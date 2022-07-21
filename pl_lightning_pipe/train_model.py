#https://www.kaggle.com/code/kickitlikeshika/bert-for-sentiment-analysis-5th-place-solution

# %%

from clearml import Task
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer # type: ignore
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
)
from pipe_conf import PROJECT_NAME
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, DataLoader, Sampler
from torchmetrics import Accuracy, ConfusionMatrix # type: ignore

import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, AutoTokenizer # type: ignore


# from torch.utils.tensorboard import SummaryWriter

# Task.add_requirements('requirements.txt')
# Task.add_requirements('requirements.txt')
# task = Task.init(project_name=PROJECT_NAME, 
#                 task_name='train_model',
#                 task_type='training', #type: ignore 
#                 )

# task.execute_remotely('GPU')
parameters = {
        'validation_split': 0.1,
        'seed': 42,
        'pre_trained_model': 'bert-base-uncased',
        'batch_size': 32,
        'max_length': 128,
        'lr': 2e-5,
        'num_epochs': 3,
        'accelerator': 'auto',
        'devices': 'auto',
    }

# task.connect(parameters)

class TokenizeDataset(Dataset):
    def __init__(self, df, max_len, model_str, eval=False):
        self.max_len = max_len
        self.eval = eval
        self.text = df['text'].tolist()
        if not self.eval:
            self.labels = df['label'].values
            
        tokenizer = AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
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
            return (input_ids, attention_mask)

        sentiments = self.labels[i]
        return (input_ids, attention_mask, sentiments)
    
    def __len__(self):
        return len(self.text)

class StratifiedSampler(Sampler):
    def __init__(self, y, batch_size):
      self.y = y
      self.batch_size = batch_size
      self.n_splits = int(np.ceil(len(y) / batch_size))
      self.n_classes = len(np.unique(y))
      self.n_samples_per_split_per_class = int(np.ceil(len(y) / self.n_splits / self.n_classes))

    def __iter__(self):
      pass

    def __len__(self):
        return len(self.y)

class DataModule(pl.LightningDataModule):
  pass

class ModBertBase(pl.LightningModule):
    
    def __init__(self, params, head_dropout = 0.2, num_classes = 5):
        super().__init__()
        self.learning_rate = params['lr']
        self.max_seq_len = params['max_length']
        self.batch_size = params['batch_size']
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.model_str = params['pre_trained_model']
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(self.model_str)
        self.pretrain_model  = AutoModel.from_pretrained(self.model_str, self.config)

        self.hidden_dim = 1024 if 'large' in self.model_str.split('-') else 768

        # The fine-tuning model head:
        layers = []
        layers.append(nn.Linear(self.hidden_dim, self.hparams.num_classes)) # type: ignore
        layers.append(nn.Dropout(self.hparams.head_dropout)) # type: ignore
        layers.append(nn.LogSoftmax(dim=1))
        self.new_layers = nn.Sequential(*layers)

    def prepare_data(self):
      tokenizer = AutoTokenizer.from_pretrained(self.model_str, trust_remote_code=True, use_fast=True) # type: ignore

      tokens_train = tokenizer.batch_encode_plus(
          x_train.tolist(),
          padding='max_length',
          max_length = self.max_seq_len,
          truncation=True,
          return_token_type_ids=False,
          return_attention_mask=True,
      )

      tokens_val = tokenizer.batch_encode_plus(
          x_val.tolist(),
          padding='max_length',
          max_length = self.max_seq_len,
          truncation=True,
          return_token_type_ids=False,
          return_attention_mask=True
      )

      tokens_test = tokenizer.batch_encode_plus(
          x_test.tolist(),
          padding='max_length',
          max_length = self.max_seq_len,
          truncation=True,
          return_token_type_ids=False,
          return_attention_mask=True
      )

      self.train_seq = torch.tensor(tokens_train['input_ids'])
      self.train_mask = torch.tensor(tokens_train['attention_mask'])
      self.train_y = torch.tensor(y_train.tolist())

      self.val_seq = torch.tensor(tokens_val['input_ids'])
      self.val_mask = torch.tensor(tokens_val['attention_mask'])
      self.val_y = torch.tensor(y_val.tolist())

      self.test_seq = torch.tensor(tokens_test['input_ids'])
      self.test_mask = torch.tensor(tokens_test['attention_mask'])
      self.test_y = torch.tensor(y_test.tolist())

    def forward(self, encode_id, mask): 
        outputs = self.pretrain_model(encode_id, attention_mask=mask)
        return self.new_layers(outputs.pooler_output)

    def train_dataloader(self):
      train_dataset = TensorDataset(self.train_seq, self.train_mask, self.train_y)
      self.train_dataloader_obj = DataLoader(train_dataset, batch_size=self.batch_size)
      return self.train_dataloader_obj

    def val_dataloader(self):
      test_dataset = TensorDataset(self.test_seq, self.test_mask, self.test_y)
      self.test_dataloader_obj = DataLoader(test_dataset, batch_size=self.batch_size)
      return self.test_dataloader_obj

    def test_dataloader(self):
      test_dataset = TensorDataset(self.test_seq, self.test_mask, self.test_y)
      self.test_dataloader_obj = DataLoader(test_dataset, batch_size=self.batch_size)
      return self.test_dataloader_obj

    def training_step(self, batch, batch_idx):
      encode_id, mask, targets = batch

      outputs = self(encode_id, mask) 
      preds = torch.argmax(outputs, dim=1)
      train_accuracy = self.accuracy(preds, targets)
      loss = self.loss(outputs, targets)

      self.log('train_accuracy', train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
      self.log('train_loss', loss, on_step=False, on_epoch=True)
      return {"loss":loss, 'train_accuracy': train_accuracy}

    def validation_step(self, batch, batch_idx):
      encode_id, mask, targets = batch
      outputs = self.forward(encode_id, mask)
      preds = torch.argmax(outputs, dim=1)
      val_accuracy = self.accuracy(preds, targets)
      loss = self.loss(outputs, targets)
      self.log("val_accuracy", val_accuracy, prog_bar = True, on_step = True, on_epoch=True)
      self.log("val_loss", loss, on_step = True, on_epoch=True)
      return {"val_loss":loss, "val_accuracy": val_accuracy}
    
    def test_step(self, batch, batch_idx):
      encode_id, mask, targets = batch
      outputs = self.forward(encode_id, mask)
      preds = torch.argmax(outputs, dim=1)
      test_accuracy = self.accuracy(preds, targets)
      loss = self.loss(outputs, targets)
      return {"test_loss":loss, "test_accuracy":test_accuracy, "preds":preds, "targets":targets}

    def test_epoch_end(self, outputs):
      test_outs = []
      for test_out in outputs:
          out = test_out['test_accuracy'] # type: ignore
          test_outs.append(out)
      
      total_test_accuracy = torch.stack(test_outs).mean()
      self.log('total_test_accuracy', total_test_accuracy, on_step=False, on_epoch=True)

      return total_test_accuracy

    def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

# if __name__ == '__main__':

# %%

#Grabs the preprocessed data from the previous step:
preprocess_task = Task.get_task(task_name='data_split',
                              project_name=PROJECT_NAME)


Path('data/interim').mkdir(parents=True, exist_ok=True)

train_data = preprocess_task.artifacts['train_data'].get()
test_data = preprocess_task.artifacts['test_data'].get()
valid_data = preprocess_task.artifacts['validation_data'].get()

# train_data = pd.read_csv(train_path)
# test_data = pd.read_csv(test_path)
# valid_data = pd.read_csv(valid_path)

local_data_path = Path(os.getcwd()) / 'data' 

# # #Defines training callbacks.
model_name = parameters['pre_trained_model']
model_path = local_data_path / 'models' / f'{model_name}'
# model_path.mkdir(parents=True, exist_ok=True)

# checkpoint_callback = ModelCheckpoint(
#     monitor='val_loss',
#     dirpath=str(model_path)
#     )

# #Trains the model.
# x_train, x_val, x_test = train_data['text'], valid_data['text'], test_data['text']
# y_train, y_val, y_test = train_data['label'], valid_data['label'], test_data['label']


# model = BertBase(params=parameters)

# # # # model = train_model(dm, parameters)
# trainer = pl.Trainer(max_epochs=parameters['num_epochs'], accelerator=parameters['accelerator'], 
#                      devices=parameters['devices'], logger=True)

# trainer.fit(model)
# trainer.save_checkpoint(f"{model_name}.ckpt")

# # #Stores the trained model as an artifact (zip).:w
# task.upload_artifact(checkpoint_callback.best_model_path, 'model_best_checkpoint')


max_len = 128
model_str = parameters['pre_trained_model']
train_dataset = TokenizeDataset(train_data, max_len, model_str)
valid_dataset = TokenizeDataset(valid_data, max_len, model_str, eval=True)
test_dataset = TokenizeDataset(test_data, max_len, model_str, eval=True)

    
# %%
train_dataloader = DataLoader(train_dataset, batch_size=1)
val_dataloader = DataLoader(valid_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

# %%
