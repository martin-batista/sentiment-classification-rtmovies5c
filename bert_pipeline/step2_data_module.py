from clearml import Task
from pathlib import Path
import json
from transformers import AutoTokenizer # type: ignore
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
)

PROJECT_NAME = 'sentiment-classification-rtmovies5c'

task = Task.init(project_name=PROJECT_NAME, task_name='BERT_pipeline_2 data_module')

parameters = {
    'preprocess_task_id': '0b8cca4925644b43bedaa7de4cbe9831',
    'pre_trained_model': 'bert-base-uncased',
    'batch_size': 16,
    'max_length': 512,
}

task.connect(parameters)

def build_data_module(data_path, parameters):
    train_path = Path(data_path) / 'train_data.json'
    valid_path = Path(data_path) / 'valid_data.json'
    test_path = Path(data_path) / 'test_data.json'

    tokenizer = AutoTokenizer.from_pretrained(parameters['pre_trained_model'])
    return TextClassificationDataModule(batch_size=parameters['batch_size'],
                                        max_length=parameters['max_length'], 
                                        train_file=train_path,
                                        validation_file=valid_path,
                                        test_file=test_path,
                                        tokenizer=tokenizer)

def main(parameters):
    #Grabs the preprocessed data from the previous step:
    preprocess_task = Task.get_task(task_id=parameters['preprocess_task_id'])
    train_data = preprocess_task.artifacts['train_data'].get()
    valid_data = preprocess_task.artifacts['validation_data'].get()
    test_data = preprocess_task.artifacts['test_data'].get()

    #Constructs the data paths to store the train, validation and test data.
    data_path = Path(__file__).parents[1] / 'data' / 'interim'
    data_path.mkdir(parents=True, exist_ok=True)

    #Stores the train, validation and test data in the data path.
    for data, data_type in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        with open(data_path / f'{data_type}_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f)

    #Constructs the data module.
    dm = build_data_module(data_path, parameters)
    task.upload_artifact('data_module', dm)


if __name__ == '__main__':
    main(parameters)



    