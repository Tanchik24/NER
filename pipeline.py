import os
from dotenv import load_dotenv
from transformers import BertTokenizer
import torch

from src.data.RawDataPreprocessor import raw_data_preprocessor
from src.features.FeaturesPreprocessor import FeaturesPreprocessor
from src.features.feature_utils import make_dataloader

load_dotenv()


def prepare_dataloader(tokenizer, texts, entities_labels, stage):
    if os.path.exists(os.path.join(os.getenv('PREPARED_DATA_FOLDER'), stage)):
        features = torch.load(os.path.join(os.getenv('PREPARED_DATA_FOLDER')), stage)
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


pipeline()
