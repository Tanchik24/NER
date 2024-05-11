import os
from dotenv import load_dotenv

load_dotenv()


def get_entities():
    with open(os.path.join(os.getenv('PROCESSED_DIR_PATH'), os.getenv('ENTITIES_FILE_NAME'))) as f:
        entities = [entity.replace('\n', '') for entity in f.readlines()]
        return entities
