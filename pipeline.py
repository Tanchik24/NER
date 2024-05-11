import os
from typing import List
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
import torch

from src.data.RawDataPreprocessor import raw_data_preprocessor
from src.features.FeaturesPreprocessor import FeaturesPreprocessor
from src.features.feature_utils import make_dataloader
from model.EntityExtractionBERT import EntityExtractionBERT

load_dotenv()


def prepare_dataloader(tokenizer: BertTokenizer, texts: List[List[str]], entities_labels: List[List[int]], stage: str) -> DataLoader:
    if os.path.exists(os.path.join(os.getenv('PREPARED_DATA_FOLDER'), stage)):
        features = torch.load(os.path.join(os.getenv('PREPARED_DATA_FOLDER'), stage))
    else:
        feature_preprocessor = FeaturesPreprocessor(tokenizer)
        features = feature_preprocessor.get_feautures(texts, entities_labels, stage)

    dataloader = make_dataloader(features, stage)
    return dataloader


def pipeline():
    texts, entities_labels = raw_data_preprocessor.preprocess_raw_data('train')
    tokenizer = BertTokenizer.from_pretrained(os.getenv('BERT_PRETRAINED'))
    if not os.path.exists(os.getenv('PREPARED_DATA_FOLDER')):
        os.makedirs(os.getenv('PREPARED_DATA_FOLDER'))

    train_dataloader = prepare_dataloader(tokenizer, texts, entities_labels, stage='train')
    train_dataloader_iter = iter(train_dataloader)
    input_ids, attention_mask, token_type_ids, slot_labels_ids = next(train_dataloader_iter)
    config = BertConfig.from_pretrained(os.environ.get('BERT_PRETRAINED'))
    model = EntityExtractionBERT(config=config)
    model(input_ids, attention_mask, token_type_ids, slot_labels_ids)


pipeline()
