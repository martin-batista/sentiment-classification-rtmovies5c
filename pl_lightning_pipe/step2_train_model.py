from clearml import Task
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer # type: ignore
from pl_transformer import ClassificationTransformer 
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
)
from pipe_conf import PROJECT_NAME
from pytorch_lightning.loggers import TensorBoardLogger
# from torch.utils.tensorboard import SummaryWriter

Task.add_requirements('requirements.txt')
task = Task.init(project_name=PROJECT_NAME, 
                   task_name='LitTransformers_pipe_2_train_model',
                   task_type='data_processing', #type: ignore 
                #  repo='https://github.com/martin-batista/sentiment-classification-rtmovies5c.git',
                #    add_task_init_call=True,
                #    requirements_file = 'requirements.txt',
                 )
# task.execute_remotely('GPU')

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

def build_data_module(path, parameters):
    tokenizer = AutoTokenizer.from_pretrained(parameters['pre_trained_model'])
    valid = parameters['validation_split'] > 0.0
    return TextClassificationDataModule(batch_size=parameters['batch_size'],
                                        max_length=parameters['max_length'], 
                                        train_file=f'{path}/train.json',
                                        validation_file= f'{path}/valid.json' if valid else None,
                                        test_file=f'{path}/test.json',
                                        tokenizer=tokenizer)


def train_model(data_module, parameters):
     return ClassificationTransformer(pretrained_model_name_or_path=parameters['pre_trained_model'],
                                       num_labels=data_module.num_classes,
                                       lr = parameters['lr'],
                                       freeze_backbone=parameters['freeze_backbone'])


def main(parameters=parameters, task=task):
    #Grabs the preprocessed data from the previous step:
    preprocess_task = Task.get_task(task_name='LitTransformers_pipe_1_data_split',
                                    project_name=PROJECT_NAME)

    train_data = preprocess_task.artifacts['train_data'].get()
    valid_data = preprocess_task.artifacts['validation_data'].get()
    test_data = preprocess_task.artifacts['test_data'].get()

    #Constructs the data paths to store the train, validation and test data.
    data_path = Path(__file__).parents[1] / 'data' 
    interim_path = data_path / 'interim'
    interim_path.mkdir(parents=True, exist_ok=True)

    #Stores the data locally for training.
    train_data.to_json(interim_path / 'train.json', orient='records', lines=True)
    valid_data.to_json(interim_path / 'valid.json', orient='records', lines=True)
    test_data.to_json(interim_path / 'test.json', orient='records', lines=True)

    #Constructs the data module.
    dm = build_data_module(interim_path, parameters)

    #Defines training callbacks.
    model_name = parameters['pre_trained_model']
    model_path = data_path / 'models' / f'{model_name}'
    model_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=str(model_path))
    

    logger = TensorBoardLogger("tb_logs", name=model_name)

    #Trains the model.
    model = train_model(dm, parameters)
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, dm)

    #Stores the trained model as an artifact (zip).
    task.upload_artifact(str(model_path), 'model')


if __name__ == '__main__':
    main()



    