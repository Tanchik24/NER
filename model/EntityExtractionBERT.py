import os
from typing import List

from transformers import BertPreTrainedModel
from model.model_utils import get_entities
from transformers import BertModel
import torch.nn as nn
import torch
from model.efficient_kan.kan import KAN


class EntityExtractionBERT(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.entities: List[str] = get_entities()
        self.num_entities: int = len(self.entities)
        self.bert: BertModel = BertModel.from_pretrained(os.getenv('BERT_PRETRAINED'))
        self.entity_classifier = KAN([config.hidden_size, int(os.getenv('BATCH_SIZE')), self.num_entities])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        batch_size, seq_len, hidden_size = sequence_output.shape
        sequence_output = sequence_output.view(batch_size * seq_len, hidden_size)

        entity_logits = self.entity_classifier(sequence_output)

        entity_logits = entity_logits.view(batch_size, seq_len,
                                           self.num_entities)

        entity_probs = torch.softmax(entity_logits, dim=-1)

        return entity_logits, entity_probs
