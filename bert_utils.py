import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel


class BertBilateralClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, self.config.num_labels)

        self.init_weights()

    def forward(self, input_1, input_2, labels=None):

        outputs_1 = self.bert(input_ids=input_1['input_ids'], 
                              attention_mask=input_1['attention_mask'], 
                              token_type_ids=input_1['token_type_ids'])
        outputs_2 = self.bert(input_ids=input_2['input_ids'], 
                              attention_mask=input_2['attention_mask'], 
                              token_type_ids=input_2['token_type_ids'])

        pooled_outputs = torch.cat((outputs_1[1], outputs_2[1]), dim=1)
        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, logits)

        return outputs  # (loss), logits
    

class BertBilateralClassification_2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, self.config.num_labels)

        self.init_weights()

    def forward(self, input_1, input_2, labels=None):

        outputs_1 = self.bert(input_ids=input_1['input_ids'], 
                              attention_mask=input_1['attention_mask'], 
                              token_type_ids=input_1['token_type_ids'])
        outputs_2 = self.bert(input_ids=input_2['input_ids'], 
                              attention_mask=input_2['attention_mask'], 
                              token_type_ids=input_2['token_type_ids'])
        
        pooled_outputs = torch.cat((outputs_1[1], outputs_2[1], 
                                    torch.abs(outputs_1[1]-outputs_2[1])), dim=1)
        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, logits)

        return outputs  # (loss), logits
    

class BertSiameseClassification_1(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, self.config.num_labels)

        self.init_weights()

    def forward(self, input_1, input_2, labels=None):

        outputs_1 = self.bert(input_ids=input_1['input_ids'], 
                              attention_mask=input_1['attention_mask'], 
                              token_type_ids=input_1['token_type_ids'])
        outputs_2 = self.bert(input_ids=input_2['input_ids'], 
                              attention_mask=input_2['attention_mask'], 
                              token_type_ids=input_2['token_type_ids'])
        
        pooled_outputs_1 = outputs_1[1]
        pooled_outputs_2 = outputs_2[1]
        
        pooled_outputs = torch.cat((pooled_outputs_1, pooled_outputs_2,
                                    torch.abs(pooled_outputs_1-pooled_outputs_2)), dim=1)
        
        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, logits)

        return outputs  # (loss), logits
    
    
class BertSiameseClassification_2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*4, self.config.num_labels)

        self.init_weights()
        
    def _mean_pooling(self, token_embeddings, input_mask):
        input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        sum_mask = input_mask_expanded.sum(1)        
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        output_vectors = sum_embeddings / sum_mask
        return output_vectors

    def forward(self, input_1, input_2, labels=None):

        outputs_1 = self.bert(input_ids=input_1['input_ids'], 
                              attention_mask=input_1['attention_mask'], 
                              token_type_ids=input_1['token_type_ids'])
        outputs_2 = self.bert(input_ids=input_2['input_ids'], 
                              attention_mask=input_2['attention_mask'], 
                              token_type_ids=input_2['token_type_ids'])
        
        pooled_outputs_1 = self._mean_pooling(outputs_1[0], input_1['attention_mask'])
        pooled_outputs_2 = self._mean_pooling(outputs_2[0], input_2['attention_mask'])
        
        pooled_outputs = torch.cat((pooled_outputs_1, pooled_outputs_2,
                                    pooled_outputs_1*pooled_outputs_2,
                                    torch.abs(pooled_outputs_1-pooled_outputs_2)), dim=1)
        
        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, logits)

        return outputs  # (loss), logits