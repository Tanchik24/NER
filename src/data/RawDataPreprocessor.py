import os
from typing import Tuple, List
import logging
import string
from dotenv import load_dotenv
import pandas as pd

from src.data.data_utils import get_unique_entities, make_file, created_processed_folders

logger = logging.getLogger(__name__)
load_dotenv()


class RawDataPreprocessor:
    def __init__(self):
        self.raw_data_path = os.getenv('RAW_DATA_PATH')
        self.processed_path = os.getenv('PROCESSED_DIR_PATH')
        created_processed_folders(self.processed_path)

    def __make_unique_entities_file(self, entities: List[str]) -> None:
        self.unique_entities = get_unique_entities(entities)
        entities_str = '\n'.join(self.unique_entities)
        make_file(os.path.join(self.processed_path), os.getenv('ENTITIES_FILE_NAME'), entities_str)

    def __get_text_entities(self, file_path: str) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(file_path)
        custom_punctuation = string.punctuation + '«—»•'
        translator = str.maketrans('', '', custom_punctuation)
        texts = [text.translate(translator).split() for text in df['1'].tolist()]
        entities = [text.split() for text in df['0'].tolist()]
        return texts, entities

    def preprocess_raw_data(self, stage: str) -> tuple[list[str], list[list[int]]]:
        texts, entities = self.__get_text_entities(os.path.join(self.raw_data_path, stage + '.csv'))

        if stage == os.getenv('TRAIN_FOLDER'):
            self.__make_unique_entities_file(entities)

        entities_labels = []
        for index, (text_list, entities_list) in enumerate(zip(texts, entities)):
            sentence_entity_label = []
            for entity in entities_list:
                sentence_entity_label.append(
                    self.unique_entities.index(
                        entity) if entity in self.unique_entities else self.unique_entities.index('UNK')
                )
            entities_labels.append(sentence_entity_label)

        entities_labels_str = [' '.join(map(str, entity)) for entity in entities_labels]
        texts_str = [' '.join(text) for text in texts]

        make_file(self.processed_path, os.path.join(stage, os.getenv('TEXT_FILE_NAME')), '\n'.join(texts_str))
        make_file(self.processed_path, os.path.join(stage, os.getenv('ENTITIES_FILE_NAME')), '\n'.join(entities_labels_str))

        return texts, entities_labels


raw_data_preprocessor = RawDataPreprocessor()
