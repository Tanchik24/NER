from model.model_utils import get_entities


def get_metrics_dict():
    metrcis_dict = {entity: {'precision': [], 'recall': [], 'f1-score': [], 'support': []} for entity in get_entities()}
    metrcis_dict['accuracy'] = []
    metrcis_dict['macro avg'] = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    metrcis_dict['weighted avg'] = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    return metrcis_dict