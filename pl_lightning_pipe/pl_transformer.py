import torch
import pytorch_lightning as pl
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationTransformer,
)
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchmetrics import Accuracy, ConfusionMatrix
import pytorch_lightning as pl
import transformers
from transformers import DistilBertModel, BertModel, BertConfig, DebertaModel, RobertaModel, DistilBertForSequenceClassification
from transformers import AutoModel, AutoConfig, AutoTokenizer, BertTokenizerFast, RobertaTokenizerFast, DistilBertTokenizerFast, DebertaTokenizerFast

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
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # automatically find the total number of steps we need
        num_training_steps, num_warmup_steps = self.compute_warmup(self.num_training_steps, num_warmup_steps=0.1)
        scheduler = transformers.get_linear_schedule_with_warmup( # type: ignore
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

class BertBase(pl.LightningModule):
    
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test,
                 max_seq_len=64, batch_size=128, learning_rate = 2e-5, lr_schedule = False,
                 model_str = 'bert-base-uncased', train_backbone = False, hidden_size = 768, head_depth = 1,
                 head_hidden_size = 768, head_dropout = 0, warmup_steps=2, num_classes = 5,
                 num_train_steps = 12):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes)
        self.model_str = model_str
        self.save_hyperparameters()

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val

        self.config = AutoConfig.from_pretrained(self.model_str)
        self.pretrain_model  = AutoModel.from_pretrained(self.model_str, self.config)

        for param in self.pretrain_model.parameters():
            param.requires_grad = False

        # The fine-tuning model head:
        layers = []
        layers.append(nn.Linear(self.hparams.hidden_size, self.hparams.num_classes))
        layers.append(nn.Dropout(0.1))
        layers.append(nn.LogSoftmax(dim=1))
        self.new_layers = nn.Sequential(*layers)

    def prepare_data(self):
      tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_str)

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
        output = self.new_layers(outputs.pooler_output)
       # print(outputs.last_hidden_state.shape)
      #  print(output.shape)
        return output

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
          out = test_out['test_accuracy']
          test_outs.append(out)
      
      total_test_accuracy = torch.stack(test_outs).mean()
      self.log('total_test_accuracy', total_test_accuracy, on_step=False, on_epoch=True)

      return total_test_accuracy

    def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    #   if self.hparams.lr_schedule:
    #     return {
    #     "optimizer": optimizer,
    #      "lr_scheduler": {
    #          "scheduler": transformers.get_linear_schedule_with_warmup(optimizer, self.hparams.warmup_steps, self.hparams.num_train_steps),
    #           "monitor": "train_loss"
    #       },
    #   }
    #   else:
      # return optimizer
    