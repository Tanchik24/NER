import os
from dotenv import load_dotenv
from typing import List, Tuple
import logging
from src.features.Features import Features
import torch

logger = logging.getLogger(__name__)
load_dotenv()


class FeaturesPreprocessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cls_token: str = tokenizer.cls_token
        self.sep_token: str = tokenizer.sep_token
        self.unk_token: str = tokenizer.unk_token
        self.pad_token_id: int = tokenizer.pad_token_id
        self.cls_token_segment_id: int = int(os.getenv('CLS_TOKEN_SEGMENT_ID'))
        self.sequence_a_segment_id: int = int(os.getenv('SEQUENCE_A_SEGMENT_ID'))
        self.max_len: int = int(os.getenv('MAX_SEQ_LENGTH'))
        self.mask_padding_with_zero: bool = True
        self.special_tokens_count = 2

    def cut_length(self, tokens: List[str], entity_labels_ids: List[int]) -> Tuple[List[str], List[int]]:
        if len(tokens) > self.max_len - self.special_tokens_count:
            tokens = tokens[:(self.max_len - self.special_tokens_count)]
            entity_labels_ids = entity_labels_ids[:(self.max_len - self.special_tokens_count)]
        return tokens, entity_labels_ids

    def add_sep_token(self, tokens: List[str], entity_labels_ids: List[int]) -> Tuple[List[str], List[int], List[int]]:
        tokens += [self.sep_token]
        entity_labels_ids += [self.pad_token_id]
        token_type_ids = [self.sequence_a_segment_id] * len(tokens)
        return tokens, entity_labels_ids, token_type_ids

    def add_cls_token(self, tokens: List[str], entity_labels_ids: List[int], token_type_ids: List[int]) -> Tuple[
        List[str], List[int], List[int]]:
        tokens = [self.cls_token] + tokens
        entity_labels_ids = [self.pad_token_id] + entity_labels_ids
        token_type_ids = [self.cls_token_segment_id] + token_type_ids
        return tokens, entity_labels_ids, token_type_ids

    def add_attention_mask(self, input_ids: List[int], token_type_ids: List[int], entity_labels_ids: List[int]):
        attention_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)
        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + ([self.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([self.pad_token_id] * padding_length)
        entity_labels_ids = entity_labels_ids + ([self.pad_token_id] * padding_length)

        return input_ids, attention_mask, token_type_ids, entity_labels_ids

    def get_feautures(self, texts: List[List[str]], entities_list: List[List[int]], stage: str) -> List[Features]:
        features = []
        for index, (text, entities) in enumerate(zip(texts, entities_list)):
            tokens = []
            entity_labels_ids = []
            for word, entity in zip(text, entities):
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [self.unk_token]
                tokens.extend(word_tokens)
                entity_label_id = [int(entity)] + [self.pad_token_id] * (len(word_tokens) - 1)
                entity_labels_ids.extend(entity_label_id)

            tokens, entity_labels_ids = self.cut_length(tokens, entity_labels_ids)
            tokens, entity_labels_ids, token_type_ids = self.add_sep_token(tokens, entity_labels_ids)
            tokens, entity_labels_ids, token_type_ids = self.add_cls_token(tokens, entity_labels_ids, token_type_ids)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids, attention_mask, token_type_ids, entity_labels_ids = self.add_attention_mask(input_ids,
                                                                                                   token_type_ids,
                                                                                                   entity_labels_ids)
            assert len(input_ids) == self.max_len, "Error with input length {} vs {}".format(len(input_ids),
                                                                                             self.max_len)
            assert len(attention_mask) == self.max_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), self.max_len)
            assert len(token_type_ids) == self.max_len, "Error with token type length {} vs {}".format(
                len(token_type_ids),
                self.max_len)
            assert len(entity_labels_ids) == self.max_len, "Error with slot labels length {} vs {}".format(
                len(entity_labels_ids), self.max_len)

            features.append(Features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                entity_labels_ids=entity_labels_ids))

        torch.save(features, os.path.join(os.getenv('PREPARED_DATA_FOLDER'), stage))
        return features


