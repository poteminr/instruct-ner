import re
from collections import defaultdict
import pandas as pd
from utils.instruct_utils import MODEL_INPUT_TEMPLATE


def split_entities_by_words(dct: dict[str, str]) -> dict[str, str]:
    word_lists = {}
    for key in dct:
        word_lists[key] = [word for item in dct[key] for word in item.split()]
    return word_lists


def extract_classes(input_string: str, entity_types: list[str]) -> dict[str, str]:
    if input_string.endswith(":"):
        input_string += " \n"
    answer_start_idx = input_string.find(MODEL_INPUT_TEMPLATE['output_separator'])
    output_separator_length = len(MODEL_INPUT_TEMPLATE['output_separator'])
    input_string = input_string[answer_start_idx+output_separator_length+1:] # input string should start with class tag
    
    classes = { k:[] for k in entity_types}

    pattern = rf"({'|'.join(entity_types)}):\s(.*?)(?=\n\w+:\s|$)"
    matches = re.findall(pattern, input_string)

    for class_name, value in matches:
        values = value.strip().split(', ')
        if values != ['']:
            classes[class_name] = values

    return classes


def calculate_metrics(
    extracted_entities: list[dict[str, str]],
    target_entities: list[dict[str, str]],
    entity_types: list[str],
    split_entities: bool = False,
    return_only_f1: bool = False,
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
            pred_set = set(extracted[label])
            target_set = set(target[label])

            if pred_set == target_set and len(pred_set) == 0:
                tp = 1
                fp = 0
                fn = 0
            else:
                tp = len(pred_set.intersection(target_set))
                fp = len(pred_set.difference(target_set))
                fn = len(target_set.difference(pred_set))

            overall_metrics[label]['tp'] += tp
            overall_metrics[label]['fp'] += fp
            overall_metrics[label]['fn'] += fn

    results = {}
    for label in entity_types:
        tp = overall_metrics[label]['tp']
        fp = overall_metrics[label]['fp']
        fn = overall_metrics[label]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if return_only_f1:
            results[label] = {'f1': f1}
        else:
            results[label] = {'precision': precision, 'recall': recall, 'f1': f1}

    return results


def calculate_metrics_from_dataframe(
    dataframe: pd.DataFrame,
    entity_types: list[str],
    skip_empty: bool = False,
    target_col_name: str = 'target',
    extracted_col_name: str = 'extracted',
    split_entities: bool = False
) -> dict[str, dict[str, float]]:
    if skip_empty:
        empty_template = {k: [] for k in entity_types}
        dataframe = dataframe[dataframe[target_col_name] != empty_template]

    return calculate_metrics(
        dataframe[extracted_col_name].values,
        dataframe[target_col_name].values,
        entity_types,
        split_entities
        )
