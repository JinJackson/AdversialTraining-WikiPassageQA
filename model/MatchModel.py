import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel

from transformers import BertPreTrainedModel
#from parser1 import args

class BertMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        hidden_state, pooled_output = outputs[:2]
        #hidden_state:[batch_size, sequence_length, hidden_size]  bert经过12个encoder最后的输出，与输入同维度
        #pooled_output:[batch_size, hidden_size]  #bert最后输出的第一维，[cls]的输出，代表了全局信息

        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        if labels == None:
            return logits
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

            return loss, logits


class RobertaMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state, pooled_output = outputs[:2]
        #hidden_state:[batch_size, sequence_length, hidden_size]  bert经过12个encoder最后的输出，与输入同维度
        #pooled_output:[batch_size, hidden_size]  #bert最后输出的第一维，[cls]的输出，代表了全局信息

        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        if labels == None:
            return logits
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

            return loss, logits

