import os
from typing import List


def make_file(dir: str, path: str, data: str) -> None:
    with open(os.path.join(dir, path), 'w') as file:
        file.write(data)


def get_unique_entities(annot_entities: List[str]) -> List[str]:
    entities = list({word for phrase in annot_entities for word in phrase})
    entities.append('UNK')
    return entities


def created_processed_folders(base_path: str) -> None:
    train_path = os.path.join(base_path, os.getenv('TRAIN_FOLDER'))
    test_path = os.path.join(base_path, os.getenv('TEST_FOLDER'))

    for path in [base_path, train_path, test_path]:
        if not os.path.exists(path):
            os.makedirs(path)