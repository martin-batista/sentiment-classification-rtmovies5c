from collections import namedtuple
from clearml import Task
from pathlib import Path
from typing import Dict, NamedTuple
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from transformers import (
    AutoModel, #type: ignore
    AutoTokenizer, # type: ignore
    get_constant_schedule_with_warmup, # type: ignore
    get_cosine_with_hard_restarts_schedule_with_warmup, # type: ignore
    get_linear_schedule_with_warmup # type: ignore
) 

from torch.utils.tensorboard import SummaryWriter # type: ignore

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, DataLoader, Sampler
from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix # type: ignore

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, EarlyStopping # type: ignore
from transformers import  AutoConfig, AutoTokenizer # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from pipe_conf import PROJECT_NAME

parameters = {
        'pre_trained_model': 'echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid',
        'batch_size': 32,
        'max_length': 128,
        'lr': 1e-5,
        'num_epochs': 8,
        'stratified_sampling': True,
        'stratified_sampling_position': 'first',
        'stratified_epochs': 8,
        'lr_schedule': 'warmup_linear', # warmup_linear, warmup_constant, warmup_cosine_restarts
        'lr_warmup': 0.5,
        'num_cycles': 4,
        'accelerator': 'auto',
        'devices': 'auto',
    }


labels = ['Custom', 'Alternate']
if parameters['stratified_sampling']: labels.append('Stratified')
if parameters['stratified_sampling_position'] == 'alternate': labels.append('Alternate')
if parameters['num_epochs'] > parameters['stratified_epochs']: labels.append('Random')

Task.add_requirements('requirements.txt')
task = Task.init(project_name=PROJECT_NAME, 
                 task_name=parameters['pre_trained_model'],
                 task_type='training', #type: ignore 
                 tags=labels
                )

task.connect(parameters)
task.execute_remotely('GPU')

class TokenizeDataset(Dataset):
    def __init__(self, df, max_len, model_str, eval=False):
        self.max_len = max_len
        self.eval = eval
        self.text = df['text'].tolist()
        if not self.eval:
            self.labels = df['label'].values
            
        tokenizer = AutoTokenizer.from_pretrained(model_str, lower=True, trust_remote_code=True, use_fast=False)
        self.encode = tokenizer.batch_encode_plus(
            self.text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True
        )
        
    def __getitem__(self, i):
        input_ids = torch.tensor(self.encode['input_ids'][i]) # type: ignore
        attention_mask = torch.tensor(self.encode['attention_mask'][i]) # type: ignore

        if self.eval:
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': -1}

        sentiments = torch.tensor(self.labels[i])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': sentiments}
    
    def __len__(self):
        return len(self.text)

class StratifiedSampler(Sampler):
    def __init__(self, y, batch_size):
      self.y = torch.tensor(y)
      self.batch_size = batch_size
      self.n_splits = int(np.ceil(len(y) / batch_size))
      self.n_classes = len(np.unique(y))
      self.n_samples_per_split_per_class = int(np.ceil(len(y) / self.n_splits / self.n_classes))

      # Create a list of indices for each class
      sorted_y, sorted_idxs = torch.sort(self.y)
      _, counts = torch.unique_consecutive(sorted_y, return_counts=True) # Return counts of elements in each class.
      self.replace = counts < self.n_samples_per_split_per_class # Replace for given class, if not enough elements.
      _class_idxs = torch.split(sorted_idxs, counts.tolist()) # Split the indices into each class.
      self._class_idxs = tuple(elem.numpy() for elem in _class_idxs) # (n_class, idxs_class_n)

    def __iter__(self):  # sourcery skip: for-append-to-extend, list-comprehension
        batch_idxs = []
        for _ in range(self.n_splits): # For each split
            for i, elem in enumerate(self._class_idxs): 
                batch_idxs.append(np.random.choice(elem, 
                                                size=self.n_samples_per_split_per_class,
                                                replace=self.replace[i] 
                                                )
                                ) # Generate stratified batch indices for each class.

        batch_idxs = np.concatenate(batch_idxs)
        np.random.shuffle(batch_idxs) # Shuffle batch indices to break sequence order e.g (0,0,0, ... 4,4,4).
        
        return iter(batch_idxs)

    def __len__(self):
        return len(self.y)

class TransformerDataModule(pl.LightningDataModule):
   
   def __init__(self, params, train_data_path, test_data_path, valid_data_path = None,
                stratified = parameters['stratified_sampling'], strat_epochs = parameters['stratified_epochs'],
                num_workers=2):
       super().__init__()
       self.params = params
       self.train_data_path = train_data_path
       self.test_data_path = test_data_path
       self.valid_data_path = valid_data_path
       self.num_epochs = self.params['num_epochs']
       self.batch_size = params['batch_size']
       self.prepare_data_per_node = False
       self.stratified = stratified
       self.strat_epochs = strat_epochs
       self.strat_pos = params['stratified_sampling_position'].split(' ')
       self.num_workers = num_workers

   def prepare_data(self):
       train_data = pd.read_csv(self.train_data_path)
       self.y = np.array(train_data['label'].values)
       self.num_classes = len(np.unique(self.y))
       test_data = pd.read_csv(self.test_data_path)
       
       self.train_tokenized = TokenizeDataset(train_data, self.params['max_length'],
                                              self.params['pre_trained_model'])
       self.test_tokenized = TokenizeDataset(test_data, self.params['max_length'],
                                             self.params['pre_trained_model'])
       if self.valid_data_path:
           valid_data = pd.read_csv(self.valid_data_path)
           self.valid_tokenized = TokenizeDataset(valid_data, self.params['max_length'],
                                                  self.params['pre_trained_model'])

   def train_dataloader(self):   # sourcery skip: lift-duplicated-conditional
       if self.stratified and 'last' in self.strat_pos:
            if self.trainer.current_epoch < (self.num_epochs - self.strat_epochs):  # type: ignore
                return DataLoader(self.train_tokenized, batch_size=self.batch_size, num_workers=self.num_workers)

       elif self.stratified and in self.strat_pos:
            if self.trainer.current_epoch <= self.strat_epochs: # type: ignore
                return DataLoader(self.train_tokenized, batch_size=self.batch_size, 
                                  sampler=StratifiedSampler(self.y, self.batch_size),
                                  num_workers=self.num_workers)
       elif 'alternate' in self.strat_pos:
                    if self.trainer.current_epoch % 2 != 0: # type: ignore
                        return DataLoader(self.train_tokenized, batch_size=self.batch_size, 
                                        sampler=StratifiedSampler(self.y, self.batch_size),
                                        num_workers=self.num_workers)
                    else:
                        return DataLoader(self.train_tokenized, batch_size=self.batch_size, 
                                        num_workers=self.num_workers) 

       return DataLoader(self.train_tokenized, batch_size=self.batch_size, 
                                num_workers=self.num_workers)
   
   def val_dataloader(self):
      return DataLoader(self.valid_tokenized, batch_size=self.batch_size, 
                        num_workers=self.num_workers)
    
   def test_dataloader(self):
      return DataLoader(self.test_tokenized, batch_size=self.batch_size,
                        num_workers=self.num_workers)

   def predict_dataloader(self):
      #Use validation data for predictions.
      return DataLoader(self.valid_tokenized, batch_size=self.batch_size,
                        num_workers=self.num_workers)    

class TransformerBase(pl.LightningModule):
    
    def __init__(self, params, head_dropout = 0.2, num_classes = 5, hidden_dim = None):
        super().__init__()
        self.learning_rate = params['lr']
        self.max_seq_len = params['max_length']
        self.batch_size = params['batch_size']
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.model_str = params['pre_trained_model']
        self.num_epochs = params['num_epochs']
        self.lr_schedule = params['lr_schedule']
        self.warmup_steps = np.ceil(params['lr_warmup']*self.num_epochs)
        self.num_cycles = params['num_cycles']

        self.save_hyperparameters()
        self.configure_metrics()

        config = AutoConfig.from_pretrained(self.model_str, num_labels=num_classes)
        self.backbone = AutoModel.from_config(config)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.config.hidden_size, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        if hidden_dim is None:
            self.hidden_dim = 1024 if 'large' in self.model_str.split('-') else 768
        else:
            self.hidden_dim = hidden_dim
    
    def model(self, input_ids, attention_mask, labels=None) -> NamedTuple:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        preds = self.classifier(outputs.pooler_output)
        loss = self.loss(preds, labels)
        results = namedtuple('ModelOutputs', ['loss', 'preds', 'last_hidden_state'])
        return results(loss=loss, preds=preds, last_hidden_state=outputs.last_hidden_state)
    
    def forward(self, input_ids, attention_mask) -> NamedTuple:
        return model(input_ids, attention_mask=attention_mask)

    def configure_metrics(self) -> None:
        self.prec = Precision(num_classes=self.num_classes, average="macro")
        self.recall = Recall(num_classes=self.num_classes, average="macro")
        self.acc = Accuracy()
        self.metrics = {"precision": self.prec, "recall": self.recall, "accuracy": self.acc}

    def compute_metrics(self, preds, labels, mode="val") -> Dict[str, torch.Tensor]:
        # Not required by all models. Only required for classification
        return {f"{mode}_{k}": metric(preds, labels) for k, metric in self.metrics.items()}

    def common_step(self, prefix: str, batch) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss # type: ignore
        preds = outputs.preds # type: ignore
        if batch["labels"] is not None:
            metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
            self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        return  loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.common_step("train", batch)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self.common_step("val", batch)

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        if -1 in batch["labels"]:
            batch["labels"] = None
        return self.common_step("test", batch)

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.model(**batch)
        return outputs.preds # type: ignore

    def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), self.learning_rate)
      if self.lr_schedule == 'warmup_linear':
        return {
        "optimizer": optimizer,
         "lr_scheduler": {
             "scheduler": get_linear_schedule_with_warmup(optimizer, self.warmup_steps, self.num_epochs),
             "name": "warmup_linear",
             "monitor": "train_loss"
          },
      }
      elif self.lr_schedule == 'warmup_cosine_restarts':
        return {
        "optimizer": optimizer,
         "lr_scheduler": {
             "scheduler": get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, self.warmup_steps, 
                                                                             self.num_epochs, self.num_cycles),
             "name": 'warmup_cosine_restarts',
             "monitor": "train_loss"
          },
      }
      elif self.lr_schedule == 'warmup_constant':
        return {
        "optimizer": optimizer,
         "lr_scheduler": {
             "scheduler": get_constant_schedule_with_warmup(optimizer, self.warmup_steps),
             "name": 'warmup_constant',
             "monitor": "train_loss"
          },
      }
      else:
        return optimizer


if __name__ == '__main__':

    #Grabs the preprocessed data from the previous step:
    preprocess_task = Task.get_task(task_name='data_split',
                                    project_name=PROJECT_NAME)

    pre_conf = preprocess_task.get_parameters_as_dict(cast=True)
    task.connect(pre_conf['General'])

    pl.seed_everything(task.get_parameter('General/seed')) # type: ignore

    Path('../data/interim').mkdir(parents=True, exist_ok=True)
    train_data_path = preprocess_task.artifacts['train_data'].get_local_copy()
    test_data_path = preprocess_task.artifacts['test_data'].get_local_copy()
    try:
        valid_data_path = preprocess_task.artifacts['validation_data'].get_local_copy()
    except Exception:
        valid_data_path = None

    # # #Defines trainincallbacks.
    Path('../data/models/train_callbacks').mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
    dirpath='../data/models/train_callbacks',
    save_top_k=2,
    monitor='val_loss',
    mode='min',
    save_weights_only=True,
    filename='{epoch}-{val_loss:.2f}'
    )

    # #Trains the model.
    model = TransformerBase(params=parameters)

    dm = TransformerDataModule(parameters, train_data_path, test_data_path, valid_data_path)
    trainer = pl.Trainer(max_epochs=parameters['num_epochs'], 
                        accelerator='gpu', 
                        devices=parameters['devices'], 
                        logger=True,
                        callbacks=[TQDMProgressBar(refresh_rate=300),
                                   LearningRateMonitor(logging_interval='epoch', log_momentum=True),
                                   EarlyStopping(monitor="val_loss", mode="min", patience=2),
                                   checkpoint_callback])

    trainer.fit(model, dm)

    #Test the model.
    trainer.test(model, dm)

    # Confusion matrix plot:
    preds = trainer.predict(model, dm)

    labels = [batch['labels'] for batch in dm.predict_dataloader()]
    labels = torch.cat(labels)
    preds = torch.cat(preds)  # type: ignore

    cm = ConfusionMatrix(num_classes=5, normalize='true')
    conf_mat = cm(preds, labels)
    acc = Accuracy()
    pred_accuracy = acc(preds, labels)
    task.get_logger().report_single_value('Prediction accuracy', pred_accuracy)

    df_cm = pd.DataFrame(conf_mat.cpu().numpy(), index = range(model.num_classes), columns=range(model.num_classes))
    plt.figure(figsize = (10,8))
    fig_ = sns.heatmap(df_cm, annot=True, linewidths=.5, cmap="YlGnBu", fmt='.2f').get_figure()  # type: ignore
    plt.close(fig_)

    task.get_logger().report_matplotlib_figure("Confusion matrix", "Validation data", fig_)
