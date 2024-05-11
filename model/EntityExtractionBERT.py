import os
from typing import List

from transformers import BertPreTrainedModel
from model.model_utils import get_entities
from transformers import BertModel
import torch.nn as nn
import torch


class EntityClassifier(nn.Module):
    def __init__(self, input_dim: int, num_entity_labels: int, dropout_rate: float = 0.):
        super(EntityClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_entity_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class EntityExtractionBERT(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.entities: List[str] = get_entities()
        self.num_entities: int = len(self.entities)
        self.bert: BertModel = BertModel.from_pretrained(os.getenv('BERT_PRETRAINED'))
        self.entity_classifier: EntityClassifier = EntityClassifier(config.hidden_size, self.num_entities, float(os.getenv('DROPOUT_RATE')))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        entity_logits = self.entity_classifier(sequence_output)
        entity_probs = torch.softmax(entity_logits, dim=-1)
        return entity_logits, entity_probs
