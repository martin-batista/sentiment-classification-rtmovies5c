from clearml import Task, Dataset
from pathlib import Path
import os
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer # type: ignore
from pl_transformer import ClassificationTransformer, BertBase 
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
)
from pipe_conf import PROJECT_NAME
from pytorch_lightning.loggers import TensorBoardLogger
import logging
# from torch.utils.tensorboard import SummaryWriter

# Task.add_requirements('requirements.txt')

def build_data_module(train_path, test_path, valid_path, parameters):
    tokenizer = AutoTokenizer.from_pretrained(parameters['pre_trained_model'])
    valid = parameters['validation_split'] > 0
    return TextClassificationDataModule(batch_size=parameters['batch_size'],
                                        max_length=parameters['max_length'], 
                                        train_file=train_path,
                                        validation_file= valid_path if valid else None,
                                        test_file=test_path,
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
                    task_name='train_model',
                    task_type='training', #type: ignore 
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
    task.execute_remotely('default')

    #Grabs the preprocessed data from the previous step:
    preprocess_task = Task.get_task(task_name='data_split',
                                    project_name=PROJECT_NAME)


    Path('data/interim').mkdir(parents=True, exist_ok=True)

    train_data = preprocess_task.artifacts['train_data'].get()
    test_data = preprocess_task.artifacts['test_data'].get()
    valid_data = preprocess_task.artifacts['validation_data'].get()

    
    # dataset_path = Dataset.get(
    #         dataset_project=PROJECT_NAME,
    #         dataset_name='data_split'
    # ).get_local_copy()
    # print(train_data)



    # train_path = list(Path(dataset_path).glob('train.json'))[0]
    # valid_path = list(Path(dataset_path).glob('valid.json'))[0]
    # test_path = list(Path(dataset_path).glob('test.json'))[0]

    # local_data_path = Path(os.getcwd()) / 'data' 
    # local_interim_data_path = local_data_path / 'interim'
    # local_interim_data_path.mkdir(parents=True, exist_ok=True)

    # shutil.move(train_path, local_interim_data_path / 'train.json')
    # shutil.move(valid_path, local_interim_data_path / 'valid.json')
    # shutil.move(test_path, local_interim_data_path / 'test.json')

    # print(os.listdir(os.getcwd()))
    # print(os.listdir(local_data_path))
    # print(os.listdir(local_interim_data_path))

    # #Constructs the data paths to store the train, validation and test data.
    # data_path = Path(__file__).parents[1] / 'data' 
    # interim_path = data_path / 'interim'
    # interim_path.mkdir(parents=True, exist_ok=True)
    # # logging.warning(f'Saving data to {interim_path}')

    # #Stores the data locally for training.
    # train_data.to_json(interim_path / 'train.json', orient='records', lines=True)
    # valid_data.to_json(interim_path / 'valid.json', orient='records', lines=True)
    # test_data.to_json(interim_path / 'test.json', orient='records', lines=True)

    # # # Constructs the data module.
    # data_path = Path('data/interim')
    # dm = build_data_module(train_path=str(data_path / 'train_data.json'), 
    #                        valid_path=str(data_path / 'valid_data.json'),
    #                        test_path=str(data_path / 'test_data.json'),
    #                        parameters=parameters)
    # print(dm.num_classes)

    # # #Defines training callbacks.
    model_name = parameters['pre_trained_model']
    # model_path = local_data_path / 'models' / f'{model_name}'
    # model_path.mkdir(parents=True, exist_ok=True)

    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath=str(model_path))
    

    # #Trains the model.
    x_train, x_val, x_test = train_data['text'], valid_data['text'], test_data['text']
    y_train, y_val, y_test = train_data['label'], valid_data['label'], test_data['label']
    model = BertBase(x_train, y_train, x_val, y_val, x_test, y_test, model_str=model_name)

    # model = train_model(dm, parameters)
    trainer = pl.Trainer( max_epochs=parameters['num_epochs'], logger=True)
    trainer.fit(model)
    trainer.save_checkpoint(f"{model_name}.ckpt")

    # #Stores the trained model as an artifact (zip).
    # task.upload_artifact(str(model_path), 'model')


if __name__ == '__main__':
    main()



    