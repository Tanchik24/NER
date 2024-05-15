import os
from typing import List
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch

from src.data.RawDataPreprocessor import raw_data_preprocessor
from src.features.FeaturesPreprocessor import FeaturesPreprocessor
from src.features.feature_utils import make_dataloader
from src.model.train_model import Trainer

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
    texts_train, entities_labels_train = raw_data_preprocessor.preprocess_raw_data('train')
    texts_test, entities_labels_test = raw_data_preprocessor.preprocess_raw_data('test')
    tokenizer = BertTokenizer.from_pretrained(os.getenv('BERT_PRETRAINED'))
    if not os.path.exists(os.getenv('PREPARED_DATA_FOLDER')):
        os.makedirs(os.getenv('PREPARED_DATA_FOLDER'))

    train_dataloader = prepare_dataloader(tokenizer, texts_train, entities_labels_train, stage='train')
    test_dataloader = prepare_dataloader(tokenizer, texts_test, entities_labels_test, stage='test')
    trainer = Trainer(train_dataloader, test_dataloader)
    trainer.train()


pipeline()
