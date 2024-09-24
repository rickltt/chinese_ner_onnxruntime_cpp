import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from utils import valid_sequence_output
from crf import CRF

class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.loss_type = nn.CrossEntropyLoss()
        self.post_init()

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            # valid_mask,
            label_ids,
            mode
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        # sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # active_loss = attention_mask.view(-1) == 1
        # active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
        # active_labels = label_ids.contiguous().view(-1)[active_loss]

        loss = self.loss_type(logits.view(-1, self.num_labels), label_ids.view(-1))
        # loss = self.loss_type(active_logits, active_labels)
        if mode == 'train':
            return loss
        else:
            return logits, loss
        

class BertForNERCRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = nn.CrossEntropyLoss()
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            label_ids,
            mode
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        
        labels = torch.where(label_ids >= 0, label_ids, torch.zeros_like(label_ids))
        loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask)
        if mode == 'train':
            return loss
        else:
            tags = self.crf.decode(logits, attention_mask)
            return tags, loss