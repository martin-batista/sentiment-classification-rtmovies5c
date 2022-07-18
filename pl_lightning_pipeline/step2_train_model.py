from clearml import Task
from pathlib import Path
import pytorch_lightning as pl
import json
from transformers import AutoTokenizer # type: ignore
from pl_transformer import ClassificationTransformer 
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
)

PROJECT_NAME = 'sentiment-classification-rtmovies5c'

# task = Task.init(project_name=PROJECT_NAME, task_name='LitTransformers_pipe_2 - train_model')

parameters = {
    'dataset_id': 'd4eaca0579de44d5a962e6cb412cdcf8',
    'validation_split': 0.1,
    'seed': 42,
    'pre_trained_model': 'bert-base-uncased',
    'batch_size': 1,
    'max_length': 512,
    'lr': 2e-5,
    'freeze_backbone': True,
    'num_epochs': 1,
    'accelerator': 'auto',
    'devices': 'auto',
}

# task.connect(parameters)

def build_data_module(data_path, parameters):
    tokenizer = AutoTokenizer.from_pretrained(parameters['pre_trained_model'])
    return TextClassificationDataModule(batch_size=parameters['batch_size'],
                                        max_length=parameters['max_length'], 
                                        train_file=f'{data_path}/train_data.json',
                                        validation_file=f'{data_path}/valid_data.json',
                                        test_file=f'{data_path}/test_data.json',
                                        tokenizer=tokenizer)


def train_model(data_module, parameters):
     return ClassificationTransformer(pretrained_model_name_or_path=parameters['pre_trained_model'],
                                       num_labels=data_module.num_classes,
                                       lr = parameters['lr'],
                                       freeze_backbone=parameters['freeze_backbone'])


def main(parameters):
    #Grabs the preprocessed data from the previous step:
    preprocess_task = Task.get_task(task_name='LitTransformers_pipe_1 - train/val split',
                                    project_name=PROJECT_NAME)

    train_data = preprocess_task.artifacts['train_data'].get()
    valid_data = preprocess_task.artifacts['validation_data'].get()
    test_data = preprocess_task.artifacts['test_data'].get()

    #Constructs the data paths to store the train, validation and test data.
    data_path = Path(__file__).parents[1] / 'data' / 'interim'
    data_path.mkdir(parents=True, exist_ok=True)

    #Stores the data locally for training.
    train_data.to_json(data_path / 'train.json', orient='records', lines=True)
    valid_data.to_json(data_path / 'valid.json', orient='records', lines=True)
    test_data.to_json(data_path / 'valid.json', orient='records', lines=True)

    #Constructs the data module.
    dm = build_data_module(str(data_path), parameters)

    #Defines training callbacks.


    #Trains the model.
    model = train_model(dm, parameters)
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)
    # trainer.fit(model, dm)

if __name__ == '__main__':
    main(parameters)



    