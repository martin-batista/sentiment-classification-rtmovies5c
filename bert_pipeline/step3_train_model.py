import pytorch_lightning as pl
import torch
from lightning_transformers.task.nlp.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)

PROJECT_NAME = 'sentiment-classification-rtmovies5c'

parameters = {
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

def train_model(data_module, parameters):
     return ClassificationTransformer(pretrained_model_name_or_path=parameters['pre_trained_model'],
                                       num_labels=data_module.num_classes,
                                       lr = parameters['lr'],
                                       freeze_backbone=parameters['freeze_backbone'])


def main(parameters):
    


