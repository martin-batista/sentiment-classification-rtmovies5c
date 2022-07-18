from clearml import Task
import pandas as pd
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from scipy.stats import wasserstein_distance


PROJECT_NAME = 'sentiment-classification-rtmovies5c'

task = Task.init(project_name=PROJECT_NAME, task_name='BERT_pipeline_1_train_val_split')

parameters = {
    'validation_split': 0.1,
    'seed': 42,
}

task.connect(parameters)

def get_train_test_data():
    task = Task.get_task(task_name='train_test_raw',
                         project_name=PROJECT_NAME)

    train = task.artifacts['train'].get()
    test = task.artifacts['test'].get()

    return train, test

def train_validation_split(train: pd.DataFrame, validation_split: int, task: Task):

    idxs = train['SentenceId'].unique()
    train_mask = np.random.choice(idxs, int(len(idxs)*(1- validation_split)), replace=False)
    val_mask = np.array(list(set(idxs) - set(train_mask)))

    train_data = train[train['SentenceId'].isin(train_mask)][['SentenceId', 'text', 'label']].copy()
    validation_data = train[train['SentenceId'].isin(val_mask)][['SentenceId', 'text', 'label']].copy()

    #Overlap test:
    assert not set(train_data['SentenceId'].unique()).intersection(set(validation_data['SentenceId'].unique()))

    #Wasserstein distance between distributions:
    w_distance = wasserstein_distance(train_data['label'].values, validation_data['label'].values)

    train_json = train_data[['label', 'text']].to_json(orient='records')
    valid_json = validation_data[['label', 'text']].to_json(orient='records')

    #Store the data:
    task.upload_artifact(name='train_data', artifact_object=train_json)
    task.upload_artifact(name='validation_data', artifact_object=valid_json)

    #Log data information:
    task.get_logger().report_table(title='Train examples',series='pandas DataFrame',iteration=0,table_plot=train_data)
    task.get_logger().report_table(title='Validation examples',series='pandas DataFrame',iteration=0,table_plot=validation_data)
    task.get_logger().report_single_value('Train size', len(train_data))
    task.get_logger().report_single_value('Validation size', len(validation_data))
    task.get_logger().report_single_value('Train sentences', train_data['SentenceId'].nunique())
    task.get_logger().report_single_value('Validation sentences', validation_data['SentenceId'].nunique())
    task.get_logger().report_single_value('Wasserstein distance', round(w_distance,5))

    def log_histogram(task, df_1, df_2, title, name_1, name_2):
        histogram_1 = df_1['label'].value_counts()/len(df_1)
        histogram_2 = df_2['label'].value_counts()/len(df_2)
        task.get_logger().report_histogram(
            title=title,
            series=name_1,
            values=histogram_1,
            xlabels=histogram_1.index.tolist(),
            xaxis="Class",
            yaxis="Density",
        )
        task.get_logger().report_histogram(
            title=title,
            series=name_2,
            values=histogram_2,
            xlabels=histogram_2.index.tolist(),
            xaxis="Class",
            yaxis="Density",
        )

    log_histogram(task, train_data, validation_data, title='Train/Valid splits', name_1='Train', name_2='Valid') 


def main(task=task, parameters=parameters):
    pl.seed_everything(parameters['seed'])
    train, test = get_train_test_data()
    train_validation_split(train, parameters['validation_split'], task)

    #Upload test data:
    task.upload_artifact(name='test_data', artifact_object=test.to_json(orient='records'))

if __name__ == '__main__':
    main()