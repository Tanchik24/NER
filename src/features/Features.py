import copy
from typing import List, Dict, Any
import json


class Features:
    def __init__(self, input_ids, attention_mask, token_type_ids, entity_labels_ids):
        self.input_idsL: List[int] = input_ids
        self.attention_mask: List[int] = attention_mask
        self.token_type_ids: List[int] = token_type_ids
        self.entity_labels_ids: List[int] = entity_labels_ids

    def __repr__(self) -> str:
        return str(self.to_json_string())

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
