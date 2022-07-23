from clearml import Task
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer # type: ignore

from torch.utils.tensorboard import SummaryWriter # type: ignore

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, DataLoader, Sampler
from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix # type: ignore

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from transformers import  AutoConfig, AutoTokenizer # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from pipe_conf import PROJECT_NAME

parameters = {
        'pre_trained_model': 'bert-base-uncased',
        'batch_size': 128,
        'max_length': 64,
        'lr': 2e-5,
        'num_epochs': 3,
        'stratified_sampling': False,
        'stratified_epochs': 0,
        'accelerator': 'auto',
        'devices': 'auto',
    }


labels = []
if parameters['stratified_sampling']: labels.append('Stratified')
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
       self.batch_size = params['batch_size']
       self.prepare_data_per_node = False
       self.stratified = stratified
       self.strat_epochs = strat_epochs
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

   def train_dataloader(self): 
       if not self.stratified:
           return DataLoader(self.train_tokenized, batch_size=self.batch_size, 
                                num_workers=self.num_workers)
       if self.trainer.current_epoch <= self.strat_epochs: # type: ignore
           return DataLoader(self.train_tokenized, batch_size=self.batch_size, 
                       sampler=StratifiedSampler(self.y, self.batch_size),
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

        self.save_hyperparameters()
        self.configure_metrics()

        config = AutoConfig.from_pretrained(self.model_str, num_labels=num_classes)
        self.model  = AutoModelForSequenceClassification.from_config(config)

        if hidden_dim is None:
            self.hidden_dim = 1024 if 'large' in self.model_str.split('-') else 768
        else:
            self.hidden_dim = hidden_dim
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

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
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        if batch["labels"] is not None:
            metric_dict = self.compute_metrics(preds, batch["labels"], mode=prefix)
            self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

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
        logits = outputs.logits
        return torch.argmax(logits, dim=1)

    def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


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
    valid_data_path = preprocess_task.artifacts['validation_data'].get_local_copy()

    # # #Defines training callbacks.
    model_name = parameters['pre_trained_model']

    # #Trains the model.
    model = TransformerBase(params=parameters)
    
    dm = TransformerDataModule(parameters, train_data_path, test_data_path, valid_data_path)
    trainer = pl.Trainer(max_epochs=parameters['num_epochs'], 
                        accelerator='gpu', 
                        devices=parameters['devices'], 
                        logger=True,
                        callbacks=[TQDMProgressBar(refresh_rate=2000)])

    trainer.fit(model, dm)
    trainer.save_checkpoint(f"{model_name}.ckpt")

    #Test the model.
    trainer.test(model, dm)

    # Confusion matrix plot:
    preds = trainer.predict(model, dm)

    labels = [batch['labels'] for batch in dm.predict_dataloader()]
    labels = torch.cat(labels)
    preds = torch.cat(preds)  # type: ignore

    cm = ConfusionMatrix(num_classes=5, normalize='true')
    conf_mat = cm(preds, labels)

    df_cm = pd.DataFrame(conf_mat.cpu().numpy(), index = range(model.num_classes), columns=range(model.num_classes))
    plt.figure(figsize = (10,8))
    fig_ = sns.heatmap(df_cm, annot=True, linewidths=.5, cmap="YlGnBu", fmt='.2f').get_figure()  # type: ignore
    plt.close(fig_)

    # writer = SummaryWriter()
    # writer.add_figure("Confusion matrix", fig_, model.current_epoch)
    task.get_logger().report_matplotlib_figure("Confusion matrix", "Validation data", fig_)
