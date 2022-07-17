from logging import log
from clearml import Task, TaskTypes
from clearml.automation import PipelineController, PipelineDecorator
import pytorch_lightning as pl
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)
from pathlib import Path
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer # type: ignore
import transformers

PROJECT_NAME = 'sentiment-classification-rtmovies5c'

parameters = {
    'dataset_id': '68a0d585393e407498c59c44968dcc49',
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


class ClassificationTransformer(TextClassificationTransformer): 

    def __init__(self, lr, freeze_backbone, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.freeze_backbone = freeze_backbone

    def setup(self, stage):
        # Freeze BERT backbone
        if self.freeze_backbone:
            for param in self.model.bert.parameters():
                param.requires_grad = False


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # automatically find the total number of steps we need
        num_training_steps, num_warmup_steps = self.compute_warmup(self.num_training_steps, num_warmup_steps=0.1)
        scheduler = transformers.get_linear_schedule_with_warmup( # type: ignore
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        ) 
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
 
@PipelineDecorator.component(return_values=['train_test_data'], cache=True, task_type='data_processing')
def get_train_test_data(dataset_id):
    train = Task.get_task(task_id=dataset_id).artifacts['train'].get()
    test = Task.get_task(task_id=dataset_id).artifacts['test'].get()
    return train, test


@PipelineDecorator.component(return_values=['train_val_split'], cache=True, task_type='data_processing')
def train_validation_split(train: pd.DataFrame, validation_split: float, seed: int, data_path: Path):

    np.random.seed(seed)
    idxs = train['SentenceId'].unique()
    train_mask = np.random.choice(idxs, int(len(idxs)*(1- validation_split)), replace=False)
    val_mask = np.array(list(set(idxs) - set(train_mask)))

    train_data = train[train['SentenceId'].isin(train_mask)][['SentenceId', 'text', 'label']].copy()
    validation_data = train[train['SentenceId'].isin(val_mask)][['SentenceId', 'text', 'label']].copy()

    #Overlap test:
    assert not set(train_data['SentenceId'].unique()).intersection(set(validation_data['SentenceId'].unique()))

    #Wasserstein distance between distributions:
    w_distance = wasserstein_distance(train_data['label'].values, validation_data['label'].values)

    train_data[['label', 'text']].to_json(data_path / 'train.json', orient='records', lines=True)
    validation_data[['label', 'text']].to_json(data_path / 'valid.json', orient='records', lines=True)

    PipelineDecorator.get_logger().report_table(title='Train examples',series='pandas DataFrame',iteration=0,table_plot=train_data)
    PipelineDecorator.get_logger().report_table(title='Test examples',series='pandas DataFrame',iteration=0,table_plot=validation_data)
    PipelineDecorator.get_logger().report_single_value('Train size', len(train_data))
    PipelineDecorator.get_logger().report_single_value('Test size', len(validation_data))
    PipelineDecorator.get_logger().report_single_value('Wasserstein distance', round(w_distance,5))

    def log_histogram(dataset_task, dataframe, name):
        histogram_data = dataframe['label'].value_counts()
        dataset_task.get_logger().report_histogram(
            title=name,
            series='Class distributions',
            values=histogram_data,
            iteration=0,
            xlabels=histogram_data.index.tolist(),
            yaxis='Amount of samples'
        )

    log_histogram(PipelineDecorator, train_data, 'Train') 
    log_histogram(PipelineDecorator, validation_data, 'Test') 


@PipelineDecorator.component(return_values=['data_module'], cache=True, task_type='data_processing')
def build_data_module(data_path, parameters):
    tokenizer = AutoTokenizer.from_pretrained(parameters['pre_trained_model'])
    return TextClassificationDataModule(batch_size=parameters['batch_size'],
                                        max_length=parameters['max_length'], 
                                        train_file=f'{str(data_path)}/train.json', 
                                        validation_file=f'{str(data_path)}/valid.json',
                                        tokenizer=tokenizer)

    
@PipelineDecorator.component(return_values=['model'], cache=True, task_type='training')
def train_model(data_module, parameters):
     return ClassificationTransformer(pretrained_model_name_or_path=parameters['pre_trained_model'],
                                       num_labels=data_module.num_classes,
                                       lr = parameters['lr'],
                                       freeze_backbone=parameters['freeze_backbone'])

@PipelineDecorator.pipeline(name='bert_base_pipeline', project=PROJECT_NAME, version='0.1')
def pipeline_executor(parameters):
    data_path = Path(__file__).parents[2] / 'data' / 'processed'   
    data_path.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(parameters['seed'])

    train, test = get_train_test_data(parameters['dataset_id'])
    train_validation_split(train, parameters['validation_split'], parameters['seed'], data_path)
    dm = build_data_module(data_path, parameters)
    model = train_model(dm, parameters)
    trainer = pl.Trainer(accelerator="auto", devices="auto", max_epochs=1)
    trainer.fit(model, dm)

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    pipeline_executor(parameters)

