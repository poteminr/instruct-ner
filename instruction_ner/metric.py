import re
from collections import defaultdict, Counter
import pandas as pd
from utils.instruct_utils import MODEL_INPUT_TEMPLATE


def split_entities_by_words(dct: dict[str, list[str]]) -> dict[str, list[str]]:
    word_lists = {}
    for key in dct:
        word_lists[key] = [word for item in dct[key] for word in item.split()]
    return word_lists


def extract_classes(input_string: str, entity_types: list[str]) -> dict[str, list[str]]:
    if input_string.endswith(":"):
        input_string += " \n"
    answer_start_idx = input_string.find(MODEL_INPUT_TEMPLATE['output_separator'])
    output_separator_length = len(MODEL_INPUT_TEMPLATE['output_separator'])
    input_string = input_string[answer_start_idx+output_separator_length+1:]  # input string should start with class tag

    classes = {k: [] for k in entity_types}

    pattern = rf"({'|'.join(entity_types)}):\s(.*?)(?=\n\w+:\s|$)"
    matches = re.findall(pattern, input_string)

    for class_name, value in matches:
        values = value.strip().split(', ')
        if values != ['']:
            classes[class_name] = values

    return classes


def calculate_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1
        
        
def calculate_metrics(
    extracted_entities: list[dict[str, str]],
    target_entities: list[dict[str, str]],
    entity_types: list[str],
    split_entities: bool = False,
    ignore_repetitions: bool = False
) -> dict[str, dict[str, float]]:
    assert not isinstance(extracted_entities, dict), f'expected a type list, but got {type(extracted_entities)}'
    assert not isinstance(target_entities, dict), f'expected a type list, but got {type(target_entities)}'

    overall_metrics = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in entity_types}
    for extracted, target in zip(extracted_entities, target_entities):
        if len(target.keys()) != len(extracted.keys()) and not isinstance(target, defaultdict):
            target = defaultdict(list, target)
            extracted = defaultdict(list, extracted)

        if split_entities:
            target = split_entities_by_words(target)
            extracted = split_entities_by_words(extracted)

        for label in entity_types:
            if ignore_repetitions:
                pred_set = set(extracted[label])
                target_set = set(target[label])

                tp = len(pred_set.intersection(target_set))
                fp = len(pred_set.difference(target_set))
                fn = len(target_set.difference(pred_set))
            else:
                pred_counter = Counter(extracted[label])
                target_counter = Counter(target[label])

                tp = sum((pred_counter & target_counter).values())
                fp = sum((pred_counter - target_counter).values())
                fn = sum((target_counter - pred_counter).values())
            
            overall_metrics[label]['tp'] += tp
            overall_metrics[label]['fp'] += fp
            overall_metrics[label]['fn'] += fn

    results = {}
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    for label in entity_types:
        overall_tp += overall_metrics[label]['tp']
        overall_fp += overall_metrics[label]['fp']
        overall_fn += overall_metrics[label]['fn']

        precision, recall, f1 = calculate_f1(**overall_metrics[label])
        results[label] = {'precision': precision, 'recall': recall, 'f1': f1}
        
    overall_precision, overall_recall, overall_f1 = calculate_f1(overall_tp, overall_fp, overall_fn)  
    results['overall'] = {'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1}
    return results


def calculate_metrics_from_dataframe(
    dataframe: pd.DataFrame,
    entity_types: list[str],
    skip_empty: bool = False,
    target_col_name: str = 'target',
    extracted_col_name: str = 'extracted',
    split_entities: bool = False,
    ignore_repetitions: bool = False
) -> dict[str, dict[str, float]]:
    if skip_empty:
        empty_template = {k: [] for k in entity_types}
        dataframe = dataframe[dataframe[target_col_name] != empty_template]

    return calculate_metrics(
        dataframe[extracted_col_name].values,
        dataframe[target_col_name].values,
        entity_types,
        split_entities,
        ignore_repetitions
        )


def _convert_predicted_tags_to_ids(tags: list[str], label_to_id: dict[str, int]) -> list[int]:
    return [label_to_id[tag] for tag in tags]


def _align_predicted_tags(
    tokens: list[str],
    extracted_entities: list[dict[str, str]],
    id_to_label: dict[int, str],
    convert_to_ids: bool = False
) -> list[int]:
    predicted_tags = []
    last_entity = None

    label_to_id = dict(zip(id_to_label.values(), id_to_label.keys()))

    for token in tokens:
        token_tags = []
        for entity, entity_tokens in extracted_entities.items():
            if token in entity_tokens:
                if last_entity == entity:
                    entity_tag = id_to_label[list(id_to_label.keys())[list(id_to_label.values()).index(f'I-{entity}')]]
                else:
                    entity_tag = id_to_label[list(id_to_label.keys())[list(id_to_label.values()).index(f'B-{entity}')]]
                    last_entity = entity
                token_tags.append(entity_tag)
        if not token_tags:
            token_tags.append(id_to_label[0])
            last_entity = None
        predicted_tags.extend(token_tags)

    if convert_to_ids:
        return _convert_predicted_tags_to_ids(predicted_tags, label_to_id)
    else:
        return predicted_tags