import re
from typing import Optional
import pandas as pd


def extract_classes(input_string: str) -> dict[str, str]:
    answer_start_idx = input_string.find('Ответ')
    input_string = input_string[answer_start_idx+8:]
    classes = {
        "Drugname": [],
        "Drugclass": [],
        "Drugform": [],
        "DI": [],
        "ADR": [],
        "Finding": []
    }

    pattern = r"(Drugname|Drugclass|Drugform|DI|ADR|Finding):\s(.*?)(?=\n\w+:\s|$)"
    matches = re.findall(pattern, input_string)

    for class_name, value in matches:
        values = value.strip().split(', ')
        if values != ['']:
            classes[class_name] = values

    return classes


def calculate_metrics(
    extracted_entities: list[dict[str, str]],
    target_entities: list[dict[str, str]],
    return_only_f1: bool = False,
    labels: Optional[list[str]] = None
) -> dict[str, dict[str, float]]:
    if labels is None:
        labels = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']

    overall_metrics = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in labels}
    for extracted, target in zip(extracted_entities, target_entities):
        for label in labels:
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
    for label in labels:
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
    labels: Optional[list[str]] = None,
    skip_empty: bool = False
) -> dict[str, dict[str, float]]:
    if skip_empty:
        empty_template = {'Drugname': [], 'Drugclass': [], 'Drugform': [], 'DI': [], 'ADR': [], 'Finding': []}
        dataframe = dataframe[dataframe['target'] != empty_template]

    return calculate_metrics(dataframe['extracted'].values, dataframe['target'].values, labels)
