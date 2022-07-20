#https://www.kaggle.com/code/kickitlikeshika/bert-for-sentiment-analysis-5th-place-solution

from clearml import Task
import os
from pathlib import Path
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from transformers import AutoTokenizer # type: ignore
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
)
from pipe_conf import PROJECT_NAME
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchmetrics import Accuracy, ConfusionMatrix # type: ignore
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, AutoTokenizer # type: ignore

# from torch.utils.tensorboard import SummaryWriter

# Task.add_requirements('requirements.txt')
Task.add_requirements('requirements.txt')
task = Task.init(project_name=PROJECT_NAME, 
                task_name='train_model',
                task_type='training', #type: ignore 
                )

task.execute_remotely('GPU')
parameters = {
        'validation_split': 0.1,
        'seed': 42,
        'pre_trained_model': 'bert-base-uncased',
        'batch_size': 16,
        'max_length': 128 ,
        'lr': 2e-5,
        'num_epochs': 3,
        'accelerator': 'auto',
        'devices': 'auto',
    }

task.connect(parameters)


class BertBase(pl.LightningModule):
    
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, params,
                 hidden_size = 768, head_dropout = 0, warmup_steps=2, num_classes = 5):
        super().__init__()
        self.learning_rate = params['lr']
        self.max_seq_len = params['max_length']
        self.batch_size = params['batch_size']
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes)
        self.model_str = params['pre_trained_model']
        self.save_hyperparameters()

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val

        self.config = AutoConfig.from_pretrained(self.model_str)
        self.pretrain_model  = AutoModel.from_pretrained(self.model_str, self.config)

        # The fine-tuning model head:
        layers = []
        layers.append(nn.Linear(self.hparams.hidden_size, self.hparams.num_classes)) # type: ignore
        layers.append(nn.Dropout(self.hparams.head_dropout)) # type: ignore
        layers.append(nn.LogSoftmax(dim=1))
        self.new_layers = nn.Sequential(*layers)

    def prepare_data(self):
      tokenizer = AutoTokenizer.from_pretrained(self.model_str, trust_remote_code=True) # type: ignore

      tokens_train = tokenizer.batch_encode_plus(
          self.x_train.tolist(),
          padding='max_length',
          max_length = self.max_seq_len,
          truncation=True,
          return_token_type_ids=False,
          return_attention_mask=True,
      )

      tokens_val = tokenizer.batch_encode_plus(
          self.x_val.tolist(),
          padding='max_length',
          max_length = self.max_seq_len,
          truncation=True,
          return_token_type_ids=False,
          return_attention_mask=True
      )

      tokens_test = tokenizer.batch_encode_plus(
          self.x_test.tolist(),
          padding='max_length',
          max_length = self.max_seq_len,
          truncation=True,
          return_token_type_ids=False,
          return_attention_mask=True
      )

      self.train_seq = torch.tensor(tokens_train['input_ids'])
      self.train_mask = torch.tensor(tokens_train['attention_mask'])
      self.train_y = torch.tensor(self.y_train.tolist())

      self.val_seq = torch.tensor(tokens_val['input_ids'])
      self.val_mask = torch.tensor(tokens_val['attention_mask'])
      self.val_y = torch.tensor(self.y_val.tolist())

      self.test_seq = torch.tensor(tokens_test['input_ids'])
      self.test_mask = torch.tensor(tokens_test['attention_mask'])
      self.test_y = torch.tensor(self.y_test.tolist())

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


def main():
    #Grabs the preprocessed data from the previous step:
    preprocess_task = Task.get_task(task_name='data_split',
                                    project_name=PROJECT_NAME)


    Path('data/interim').mkdir(parents=True, exist_ok=True)

    train_data = preprocess_task.artifacts['train_data'].get_local_copy()
    test_data = preprocess_task.artifacts['test_data'].get_local_copy()
    valid_data = preprocess_task.artifacts['validation_data'].get_local_copy()

    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    valid = pd.read_csv(valid_data)

    local_data_path = Path(os.getcwd()) / 'data' 
    
    # # #Defines training callbacks.
    model_name = parameters['pre_trained_model']
    model_path = local_data_path / 'models' / f'{model_name}'
    # model_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=str(model_path)
        )
    
    # #Trains the model.
    x_train, x_val, x_test = train['text'], valid['text'], test['text']
    y_train, y_val, y_test = train['label'], valid['label'], test['label']
    model = BertBase(x_train, y_train, x_val, y_val, x_test, y_test, parameters)

    # # # model = train_model(dm, parameters)
    trainer = pl.Trainer(profiler='simple', accelerator='auto')
    trainer.fit(model)
    # trainer = pl.Trainer(max_epochs=parameters['num_epochs'], accelerator=parameters['accelerator'], 
    #                      devices=parameters['devices'], logger=True, callbacks=[checkpoint_callback])
    
    # trainer.fit(model)
    # trainer.save_checkpoint(f"{model_name}.ckpt")

    # # #Stores the trained model as an artifact (zip).:w
    # task.upload_artifact(checkpoint_callback.best_model_path, 'model_best_checkpoint')


if __name__ == '__main__':
    main()



    