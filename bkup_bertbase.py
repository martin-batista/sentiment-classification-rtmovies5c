  # # class BertBase(pl.LightningModule):
      
  #     def __init__(self, params, head_dropout = 0.2, num_classes = 5):
  #         super().__init__()
  #         self.learning_rate = params['lr']
  #         self.max_seq_len = params['max_length']
  #         self.batch_size = params['batch_size']
  #         self.loss = nn.CrossEntropyLoss()
  #         self.accuracy = Accuracy()
  #         self.model_str = params['pre_trained_model']
  #         self.save_hyperparameters()

  #         self.config = AutoConfig.from_pretrained(self.model_str)
  #         self.pretrain_model  = AutoModel.from_pretrained(self.model_str, self.config)

  #         self.hidden_dim = 1024 if 'large' in self.model_str.split('-') else 768

  #         # The fine-tuning model head:
  #         layers = []
  #         layers.append(nn.Linear(self.hidden_dim, self.hparams.num_classes)) # type: ignore
  #         layers.append(nn.Dropout(self.hparams.head_dropout)) # type: ignore
  #         layers.append(nn.LogSoftmax(dim=1))
  #         self.new_layers = nn.Sequential(*layers)

  #     def prepare_data(self):
  #       tokenizer = AutoTokenizer.from_pretrained(self.model_str, trust_remote_code=True, use_fast=True) # type: ignore

  #       tokens_train = tokenizer.batch_encode_plus(
  #           x_train.tolist(),
  #           padding='max_length',
  #           max_length = self.max_seq_len,
  #           truncation=True,
  #           return_token_type_ids=False,
  #           return_attention_mask=True,
  #       )

  #       tokens_val = tokenizer.batch_encode_plus(
  #           x_val.tolist(),
  #           padding='max_length',
  #           max_length = self.max_seq_len,
  #           truncation=True,
  #           return_token_type_ids=False,
  #           return_attention_mask=True
  #       )

  #       tokens_test = tokenizer.batch_encode_plus(
  #           x_test.tolist(),
  #           padding='max_length',
  #           max_length = self.max_seq_len,
  #           truncation=True,
  #           return_token_type_ids=False,
  #           return_attention_mask=True
  #       )

  #       self.train_seq = torch.tensor(tokens_train['input_ids'])
  #       self.train_mask = torch.tensor(tokens_train['attention_mask'])
  #       self.train_y = torch.tensor(y_train.tolist())

  #       self.val_seq = torch.tensor(tokens_val['input_ids'])
  #       self.val_mask = torch.tensor(tokens_val['attention_mask'])
  #       self.val_y = torch.tensor(y_val.tolist())

  #       self.test_seq = torch.tensor(tokens_test['input_ids'])
  #       self.test_mask = torch.tensor(tokens_test['attention_mask'])
  #       self.test_y = torch.tensor(y_test.tolist())

  #     def forward(self, encode_id, mask): 
  #         outputs = self.pretrain_model(encode_id, attention_mask=mask)
  #         return self.new_layers(outputs.pooler_output)

  #     def train_dataloader(self):
  #       train_dataset = TensorDataset(self.train_seq, self.train_mask, self.train_y)
  #       self.train_dataloader_obj = DataLoader(train_dataset, batch_size=self.batch_size)
  #       return self.train_dataloader_obj

  #     def val_dataloader(self):
  #       test_dataset = TensorDataset(self.test_seq, self.test_mask, self.test_y)
  #       self.test_dataloader_obj = DataLoader(test_dataset, batch_size=self.batch_size)
  #       return self.test_dataloader_obj

  #     def test_dataloader(self):
  #       test_dataset = TensorDataset(self.test_seq, self.test_mask, self.test_y)
  #       self.test_dataloader_obj = DataLoader(test_dataset, batch_size=self.batch_size)
  #       return self.test_dataloader_obj

  #     def training_step(self, batch, batch_idx):
  #       encode_id, mask, targets = batch

  #       outputs = self(encode_id, mask) 
  #       preds = torch.argmax(outputs, dim=1)
  #       train_accuracy = self.accuracy(preds, targets)
  #       loss = self.loss(outputs, targets)

  #       self.log('train_accuracy', train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
  #       self.log('train_loss', loss, on_step=False, on_epoch=True)
  #       return {"loss":loss, 'train_accuracy': train_accuracy}

  #     def validation_step(self, batch, batch_idx):
  #       encode_id, mask, targets = batch
  #       outputs = self.forward(encode_id, mask)
  #       preds = torch.argmax(outputs, dim=1)
  #       val_accuracy = self.accuracy(preds, targets)
  #       loss = self.loss(outputs, targets)
  #       self.log("val_accuracy", val_accuracy, prog_bar = True, on_step = True, on_epoch=True)
  #       self.log("val_loss", loss, on_step = True, on_epoch=True)
  #       return {"val_loss":loss, "val_accuracy": val_accuracy}
      
  #     def test_step(self, batch, batch_idx):
  #       encode_id, mask, targets = batch
  #       outputs = self.forward(encode_id, mask)
  #       preds = torch.argmax(outputs, dim=1)
  #       test_accuracy = self.accuracy(preds, targets)
  #       loss = self.loss(outputs, targets)
  #       return {"test_loss":loss, "test_accuracy":test_accuracy, "preds":preds, "targets":targets}

  #     def test_epoch_end(self, outputs):
  #       test_outs = []
  #       for test_out in outputs:
  #           out = test_out['test_accuracy'] # type: ignore
  #           test_outs.append(out)
        
  #       total_test_accuracy = torch.stack(test_outs).mean()
  #       self.log('total_test_accuracy', total_test_accuracy, on_step=False, on_epoch=True)

  #       return total_test_accuracy

  #     def configure_optimizers(self):
  #       return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


