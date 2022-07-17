from asyncio import wait_for
from pathlib import Path
import pandas as pd
import numpy as np
from clearml import Dataset, StorageManager, Task
from scipy.stats import wasserstein_distance
from config import PROJECT_NAME, S3_BUCKET

def train_test_split(data: pd.DataFrame, test_split: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]: 
    """Generates the train and test splits.
    Args:
        data (pd.DataFrame): Data to split.
        train_split (float): Percentage of data reserved for training.
        test_split (float): Percentage of data reserved for testing.
    """
    np.random.seed(seed)
    #First split off the test data:
    idxs = data['SentenceId'].unique()
    train_size = len(idxs)*(1- test_split)
    test_size = len(idxs)*test_split

    train_mask = np.random.choice(idxs, int(train_size), replace=False)
    test_mask = np.array(list(set(idxs) - set(train_mask)))

    #Build the datasets.
    train_data = data[data['SentenceId'].isin(train_mask)][['SentenceId', 'text', 'label']].copy()
    test_data = data[data['SentenceId'].isin(test_mask)][['SentenceId', 'text', 'label']].copy()

    #Overlap test:
    assert not set(train_data['SentenceId'].unique()).intersection(set(test_data['SentenceId'].unique()))

    return train_data, test_data


def split_data(dataset_path, parameters):
    data = pd.read_csv(dataset_path, sep='\t')
    data.rename(columns={'Phrase': 'text', 'Sentiment': 'label'}, inplace=True)
    train_data, test_data = train_test_split(data, parameters['test_split'], parameters['random_seed'])
    w_distance = wasserstein_distance(train_data['label'].values, test_data['label'].values)

    return train_data, test_data, w_distance

def log_histogram(task, df_1, df_2, title, name_1, name_2):
    histogram_1 = df_1['label'].value_counts()
    histogram_2 = df_2['label'].value_counts()
    task.get_logger().report_histogram(
        title=title,
        series=name_1,
        values=histogram_1,
        xaxis="Class",
        yaxis="Density",
    )
    task.get_logger().report_histogram(
        title=title,
        series=name_2,
        values=histogram_2,
        xaxis="Class",
        yaxis="Density",
    )
   
def main():
   manager = StorageManager() 

   parameters = {
        'test_split': 0.2,
        'random_seed': 26894,
    }

   data_url = S3_BUCKET + "data/raw/data.tsv"
   metadata_url = S3_BUCKET + "data/raw/metadata.txt"

   data_path = manager.download_file(remote_url=data_url)
   metadata_path = manager.download_file(remote_url=metadata_url)
    
   # Train test split.
   train, test, w_distance = split_data(data_path, parameters)

   # Build the dataset.
   dataset = Dataset.create(
        dataset_project=PROJECT_NAME,
        dataset_name='train_test_raw',
   )
   dataset_task = Task.get_task(task_id=dataset.id)
   dataset_task.connect(parameters)

   dataset_task.get_logger().report_table(title='Train examples',series='pandas DataFrame',iteration=0,table_plot=train)
   dataset_task.get_logger().report_table(title='Test examples',series='pandas DataFrame',iteration=0,table_plot=test)
   dataset_task.get_logger().report_single_value('Train size', len(train))
   dataset_task.get_logger().report_single_value('Test size', len(test))
   dataset_task.get_logger().report_single_value('Train sentences', train['SentenceId'].nunique())
   dataset_task.get_logger().report_single_value('Validation sentences', test['SentenceId'].nunique())
   dataset_task.get_logger().report_single_value('Wasserstein distance', round(w_distance,5))

   # Log the histogram.
   log_histogram(dataset_task, train, test, title='Train/Test splits', name_1='Train', name_2='Test') 

   # Add metadata as artifact.
   dataset_task.upload_artifact(name='metadata', artifact_object=metadata_path)

   # Add train and test data.
   dataset_task.upload_artifact(name='train', artifact_object=train)
   dataset_task.upload_artifact(name='test', artifact_object=test)

   # Close the dataset and task.
   dataset.finalize()
   dataset_task.flush(wait_for_uploads=True)


if __name__ == '__main__':
    main()