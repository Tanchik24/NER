import os
from typing import List


def make_file(dir: str, path: str, data: str) -> None:
    with open(os.path.join(dir, path), 'w') as file:
        file.write(data)


def get_unique_entities(annot_entities: List[str]) -> List[str]:
    entities = list({word for phrase in annot_entities for word in phrase})
    return entities


def created_folders(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def get_data_from_file(path: str) -> list[str]:
    with open(path, 'r') as f:
        data = [line.replace('\n', '') for line in f.readlines()]
        return data
