import pytorch_lightning as pl

import torch
import torch.optim as optim

from models import ReModel, EmbeddingLayerFactories, MiddleLayerFactories, OutputLayerFactories
from metrics import calc_acc
from losses import cross_entropy_loss


class ReModelTrainer(pl.LightningModule):
    def __init__(self, config, l2i, i2l, token_pad_id):
        super().__init__()

        self.config = config
        self.l2i = l2i
        self.i2l = i2l
        self.token_pad_id = token_pad_id

        self.model = ReModel(
            embedding_layer=EmbeddingLayerFactories.get_layer[self.config.model.embedding_layer],
            middle_layer=MiddleLayerFactories.get_layer[self.config.model.middle_layer],
            output_layer=OutputLayerFactories.get_layer[self.config.model.output_layer],

            fc_in_size=self.config.parameters.fc_in_size,
            label_size=len(l2i))

    def forward(self, tokens, **parameters):
        return self.model(tokens=tokens, parameters=parameters)

    def training_step(self, batch, batch_idx):
        tokens, token_type_ids, labels = batch

        pred_y = self(
            tokens=tokens,
            token_type_ids=token_type_ids,
            attention_mask=(tokens != 1).float())

        loss = cross_entropy_loss(true_y=labels, pred_y=pred_y)
        score = calc_acc(true_y=labels.tolist(), pred_y=torch.argmax(pred_y, dim=-1).tolist())

        self.log("train_loss", loss)
        self.log("train_acc", score * 100)

        return loss

    def validation_step(self, batch, batch_idx):
        tokens, token_type_ids, labels = batch

        pred_y = self(
            tokens=tokens,
            token_type_ids=token_type_ids,
            attention_mask=(tokens != 1).float())

        loss = cross_entropy_loss(true_y=labels, pred_y=pred_y)
        score = calc_acc(true_y=labels.tolist(), pred_y=torch.argmax(pred_y, dim=-1).tolist())

        self.log("val_loss", loss)
        self.log("val_acc", score * 100)


    def configure_optimizers(self):
        # param_optimizer = list(self.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optim.AdamW(
            self.parameters(),
            lr=float(self.config.parameters.learning_rate),
            weight_decay=self.config.parameters.weight_decay)
