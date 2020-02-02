import torch
from torch import nn
import transformers


class Encoder(nn.Module) :
    def __init__(self, config) :
        super(Encoder, self).__init__()
        self.encoder = transformers.modeling_bert.BertModel.from_pretrained(config.bert_model)
        self.config = config

    def forward(self, x, x_mask) :
        _, x = self.encoder(input_ids=x, attention_mask=x_mask)
        return x

