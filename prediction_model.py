import os
import torch

from tokenizer import ReTokenizer
from models import (
    ReModel,
    EmbeddingLayerFactories,
    MiddleLayerFactories,
    OutputLayerFactories
)

from data_loader.utils import load_dump

class PredictionModel:
    def __init__(self, config, model_type):
        self.config = config
        self.model_type = model_type

        self.labels, self.l2i, self.i2l, self.tokenizer, self.model = self.load(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

    def load(self, config):
        data = load_dump(os.path.join(config.path.vocab, "labels.vocab"))
        labels = data["labels"]
        l2i = data["l2i"]
        i2l = data["i2l"]

        tokenizer = ReTokenizer(tokenizer_name=config.tokenizer.name)
        model = ReModel(
            embedding_layer=EmbeddingLayerFactories.get_layer[config.model.embedding_layer],
            middle_layer=MiddleLayerFactories.get_layer[config.model.middle_layer],
            output_layer=OutputLayerFactories.get_layer[config.model.output_layer],

            fc_in_size=config.parameters.fc_in_size,
            label_size=len(l2i))

        model_list = os.listdir(config.path.hub)
        model_fn = ""
        for model_name in model_list:
            if self.model_type in model_name:
                model_fn = model_name
                break

        model.load_state_dict(torch.load(os.path.join(config.path.hub, model_fn)))
        model.eval()

        return labels, l2i, i2l, tokenizer, model

    def forward(self, tokens, **parameters):
        pred_y = self.model(
            tokens=tokens,
            parameters=parameters)

        return pred_y

    def predict(
            self,
            sentence,

            subj_start_idx,
            subj_end_idx,
            subj_label,

            obj_start_idx,
            obj_end_idx,
            obj_label,

            return_method="label"):

        token_list, _ = self.tokenizer(
            data=[sentence, subj_start_idx, subj_end_idx, subj_label, obj_start_idx, obj_end_idx, obj_label, ""],
            method=self.config.tokenizer.marker_type
        )

        tokens = torch.tensor([self.tokenizer.convert_tokens_to_ids(["[CLS]"] + token_list)])
        token_type_ids = torch.tensor([[0] * len(tokens)])

        tokens = tokens.to(self.device)
        token_type_ids = token_type_ids.to(self.device)

        pred_y = self.forward(
            tokens=tokens,
            token_type_ids=token_type_ids,
            attention_mask=(tokens != 1).float())

        if return_method == "label":
            return self.i2l[torch.argmax(pred_y, dim=-1).item()]

        elif return_method == "prob":
            return pred_y.tolist()

        else:
            raise NotImplementedError()