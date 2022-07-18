from typing import overload
from clearml import Dataset, Task, Logger
import pandas as pd
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from scipy.stats import wasserstein_distance
from pipe_conf import PROJECT_NAME

# Task.add_requirements('requirements.txt')


def get_train_test_data(task_id):
    dataset_task = Task.get_task(task_id)

    train = dataset_task.artifacts['train'].get()
    test = dataset_task.artifacts['test'].get()

    return train, test

def train_validation_split(train: pd.DataFrame, validation_split: int):

    idxs = train['SentenceId'].unique()
    train_mask = np.random.choice(idxs, int(len(idxs)*(1- validation_split)), replace=False)
    val_mask = np.array(list(set(idxs) - set(train_mask)))

    train_data = train[train['SentenceId'].isin(train_mask)][['SentenceId', 'text', 'label']].copy()
    validation_data = train[train['SentenceId'].isin(val_mask)][['SentenceId', 'text', 'label']].copy()

    #Overlap test:
    assert not set(train_data['SentenceId'].unique()).intersection(set(validation_data['SentenceId'].unique()))

    return train_data, validation_data

def log_histogram(task, df_1, df_2, df_3, title, name_1, name_2, name_3):
    histogram_1 = df_1['label'].value_counts()/len(df_1)
    histogram_2 = df_2['label'].value_counts()/len(df_2)
    histogram_3 = df_3['label'].value_counts()/len(df_3)
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
    task.get_logger().report_histogram(
        title=title,
        series=name_3,
        values=histogram_3,
        xlabels=histogram_3.index.tolist(),
        xaxis="Class",
        yaxis="Density",
    )


def main():
    task = Task.create(project_name=PROJECT_NAME, 
                    task_name='LitTransformers_pipe_1_data_split',
                    task_type='data_processing', #type: ignore 
                    repo='https://github.com/martin-batista/sentiment-classification-rtmovies5c.git',
                    script='pl_lightning_pipe/step1_train_val_split.py',
                    add_task_init_call=True,
                    requirements_file = 'requirements.txt',
                    )

    parameters = {
        'dataset_id': '8fe0f01e7c9540ac8b94ddbc84ac7ecb',
        'validation_split': 0.1,
        'seed': 42,
    }

    task.connect(parameters)
    # task.execute_remotely('GPU')

    pl.seed_everything(parameters['seed'])
    train, test_data = get_train_test_data(parameters['dataset_id'])
    train_data, validation_data = train_validation_split(train, parameters['validation_split'])

    #Wasserstein distance between distributions:
    w_distance_train_valid = wasserstein_distance(train_data['label'].values, validation_data['label'].values)
    w_distance_train_test = wasserstein_distance(train_data['label'].values, test_data['label'].values)
    w_distance_test_valid = wasserstein_distance(test_data['label'].values, validation_data['label'].values)

    mean_w_distance = (w_distance_train_valid + w_distance_train_test + w_distance_test_valid)/3

    #Log data information:
    task.get_logger().report_table(title='Train examples',series='pandas DataFrame',iteration=0,table_plot=train_data)
    task.get_logger().report_table(title='Validation examples',series='pandas DataFrame',iteration=0,table_plot=validation_data)
    task.get_logger().report_table(title='Test examples',series='pandas DataFrame',iteration=0,table_plot=test_data)
    task.get_logger().report_single_value('Train size', len(train_data)) 
    task.get_logger().report_single_value('Validation size', len(validation_data)) 
    task.get_logger().report_single_value('Test size', len(test_data))
    task.get_logger().report_single_value('Train sentences', train_data['SentenceId'].nunique())
    task.get_logger().report_single_value('Validation sentences', validation_data['SentenceId'].nunique())  
    task.get_logger().report_single_value('Test sentences', test_data['SentenceId'].nunique())
    task.get_logger().report_single_value('Mean Wasserstein distance', round(mean_w_distance, 5))
    task.get_logger().report_single_value('Wasserstein train/valid', round(w_distance_train_valid,5))  
    task.get_logger().report_single_value('Wasserstein train/test', round(w_distance_train_test,5))  
    task.get_logger().report_single_value('Wasserstein test/valid', round(w_distance_test_valid,5)) 

    log_histogram(task, train_data, validation_data, test_data, 
                  title='Data splits', name_1='Train', 
                  name_2='Valid', name_3='Test') 

    #Store the data:
    train_save = train_data[['label', 'text']].copy()
    valid_save = validation_data[['label', 'text']].copy()
    test_save = test_data[['label', 'text']].copy()

    task.upload_artifact(name='train_data', artifact_object=train_save, wait_on_upload=True)
    task.upload_artifact(name='validation_data', artifact_object=valid_save, wait_on_upload=True)
    task.upload_artifact(name='test_data', artifact_object=test_save, wait_on_upload=True)
                  

if __name__ == '__main__':
    main()