from clearml import Task, Dataset
from pathlib import Path
import os
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer # type: ignore
from pl_transformer import ClassificationTransformer 
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
)
from pipe_conf import PROJECT_NAME
from pytorch_lightning.loggers import TensorBoardLogger
import logging
# from torch.utils.tensorboard import SummaryWriter

# Task.add_requirements('requirements.txt')

def build_data_module(parameters):
    tokenizer = AutoTokenizer.from_pretrained(parameters['pre_trained_model'])
    valid = parameters['validation_split'] > 0.0
    return TextClassificationDataModule(batch_size=parameters['batch_size'],
                                        max_length=parameters['max_length'], 
                                        train_file='data/interim/train.json',
                                        validation_file= 'data/interim/valid.json' if valid else None,
                                        test_file='data/interim/test.json',
                                        tokenizer=tokenizer)


def train_model(data_module, parameters):
     return ClassificationTransformer(pretrained_model_name_or_path=parameters['pre_trained_model'],
                                       num_labels=data_module.num_classes,
                                       lr = parameters['lr'],
                                       freeze_backbone=parameters['freeze_backbone'])


def main():
    # task = Task.create(project_name=PROJECT_NAME, 
    #                 task_name='LitTransformers_pipe_2_train_model',
    #                 task_type='data_processing', #type: ignore 
    #                 repo='https://github.com/martin-batista/sentiment-classification-rtmovies5c.git',
    #                 script='pl_lightning_pipe/step2_train_model.py',
    #                 add_task_init_call=True,
    #                 requirements_file = 'requirements.txt',
    #                 )

    Task.add_requirements('requirements.txt')
    task = Task.init(project_name=PROJECT_NAME, 
                    task_name='LitTransformers_pipe_2_train_model',
                    task_type='data_processing', #type: ignore 
                    )

    parameters = {
        'validation_split': 0.1,
        'seed': 42,
        'pre_trained_model': 'bert-base-uncased',
        'batch_size': 16,
        'max_length': 512,
        'lr': 2e-5,
        'freeze_backbone': True,
        'num_epochs': 3,
        'accelerator': 'auto',
        'devices': 'auto',
    }

    task.connect(parameters)
    task.execute_remotely('GPU')

    #Grabs the preprocessed data from the previous step:
    preprocess_task = Task.get_task(task_name='LitTransformers_pipe_1_data_split',
                                    project_name=PROJECT_NAME)

    # train_data = preprocess_task.artifacts['train_data'].get()
    # valid_data = preprocess_task.artifacts['validation_data'].get()
    # test_data = preprocess_task.artifacts['test_data'].get()
    dataset_path = Dataset.get(
            dataset_project=PROJECT_NAME,
            dataset_name='data_split'
    ).get_local_copy()


    train_path = list(Path(dataset_path).glob('train.json'))[0]
    valid_path = list(Path(dataset_path).glob('valid.json'))[0]
    test_path = list(Path(dataset_path).glob('test.json'))[0]

    local_data_path = Path(os.getcwd()) / 'data' 
    local_interim_data_path = local_data_path / 'interim'
    local_interim_data_path.mkdir(parents=True, exist_ok=True)

    shutil.move(train_path, local_interim_data_path / 'train.json')
    shutil.move(valid_path, local_interim_data_path / 'valid.json')
    shutil.move(test_path, local_interim_data_path / 'test.json')


    # #Constructs the data paths to store the train, validation and test data.
    # data_path = Path(__file__).parents[1] / 'data' 
    # interim_path = data_path / 'interim'
    # interim_path.mkdir(parents=True, exist_ok=True)
    # # logging.warning(f'Saving data to {interim_path}')

    # #Stores the data locally for training.
    # train_data.to_json(interim_path / 'train.json', orient='records', lines=True)
    # valid_data.to_json(interim_path / 'valid.json', orient='records', lines=True)
    # test_data.to_json(interim_path / 'test.json', orient='records', lines=True)

    # # Constructs the data module.
    dm = build_data_module(parameters=parameters)

    # #Defines training callbacks.
    model_name = parameters['pre_trained_model']
    model_path = local_data_path / 'models' / f'{model_name}'
    model_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=str(model_path))
    

    #Trains the model.
    model = train_model(dm, parameters)
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=parameters['num_epochs'], logger=True, enable_progress_bar=False, callbacks=[checkpoint_callback])
    trainer.fit(model, dm)
    trainer.save_checkpoint(f"{model_name}.ckpt")

    #Stores the trained model as an artifact (zip).
    task.upload_artifact(str(model_path), 'model')


if __name__ == '__main__':
    main()



    