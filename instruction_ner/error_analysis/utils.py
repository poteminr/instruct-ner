from collections import Counter, defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def aggregate_errors(
    target_datasets: list[dict[str, list[str]]],
    extracted_datasets: list[dict[str, list[str]]],
    sample_ids: list[str]
) -> dict[str]:
    overall_errors = {
        'total': 0,
        'fp': 0,
        'fn': 0,
        'mistaken_recognitions': defaultdict(list),
        'entities_not_recognized': defaultdict(list),
        'over_recognitions':defaultdict(list),
        'errors_by_keys': {},
        'number_of_entities': defaultdict(int)
    }
    for extracted_dict, target_dict, sample_id in zip(extracted_datasets, target_datasets, sample_ids):
        if not isinstance(extracted_dict, defaultdict) or not isinstance(target_dict, defaultdict):
            extracted_dict = defaultdict(list, extracted_dict)
            target_dict = defaultdict(list, target_dict)
            
        for key in extracted_dict.keys():
            extracted_values = Counter(extracted_dict[key])
            target_values = Counter(target_dict[key])
            false_positives = extracted_values - target_values
            false_negatives = target_values - extracted_values

            overall_errors['fp'] += sum(false_positives.values())
            overall_errors['fn'] += sum(false_negatives.values())

            if key not in overall_errors['errors_by_keys']:
                overall_errors['errors_by_keys'][key] = {
                    'fp': 0,
                    'fn': 0
                }

            overall_errors['errors_by_keys'][key]['fp'] += sum(false_positives.values())
            overall_errors['errors_by_keys'][key]['fn'] += sum(false_negatives.values())

            # Track mistaken recognitions with real target class
            for value in extracted_dict[key]:
                if value not in target_dict[key]:
                    real_target = next((k for k, v in target_dict.items() if value in v), None)
                    if real_target is not None:
                        overall_errors['mistaken_recognitions'][real_target].append((value, key, sample_id))
                    else:
                        overall_errors['over_recognitions'][key].append((value, sample_id))
            
            # Track entities not recognized
        for key in target_dict.keys():
            overall_errors['number_of_entities'][key] += len(target_dict[key])
            for value in target_dict[key]:
                if value not in extracted_dict[key]:
                    predicted_target = next((k for k, v in extracted_dict.items() if value in v), None)
                    if predicted_target is None:
                        overall_errors['entities_not_recognized'][key].append((value, sample_id))
                        
        overall_errors['total'] += 1
    return overall_errors


def aggregate_errors_from_dataframe(
    dataframe: pd.DataFrame,
    target_col_name: str = 'target',
    extracted_col_name: str = 'extracted',
    id_col_name: Optional[str] = 'id'
) -> dict[str]:
    if id_col_name is None or id_col_name not in dataframe.columns:
        sample_ids = np.arange(len(dataframe))
    else:
        sample_ids = dataframe[id_col_name]
        
    return aggregate_errors(dataframe[target_col_name].values, dataframe[extracted_col_name].values, sample_ids)


def plot_confusion_matrix(
    errors: dict[str],
    entity_types: Optional[list[str]] = None,
    in_percent: bool = False,
    add_description: bool = True
):
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    entity_types_with_mistakes = set()
    
    for real_target, recognitions, in errors['mistaken_recognitions'].items():
        for _, predicted_target, _ in recognitions:
                confusion_matrix[real_target][predicted_target] += 1
                entity_types_with_mistakes.add(real_target)
                entity_types_with_mistakes.add(predicted_target)

    if entity_types is None:
        entity_types = list(entity_types_with_mistakes)
    entity_types.sort()

    matrix_data = [[confusion_matrix.get(row, {}).get(col, 0) for col in entity_types] for row in entity_types]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(matrix_data, cmap='tab20b')
    ax.set_xticks(np.arange(len(entity_types)))
    ax.set_yticks(np.arange(len(entity_types)))
    ax.set_xticklabels(entity_types)
    ax.set_yticklabels(entity_types)

    for i in range(len(entity_types)):
        for j in range(len(entity_types)):
            if i != j:
                number_of_mistakes = matrix_data[i][j]
                if in_percent:
                    number_of_mistakes = number_of_mistakes / errors['number_of_entities'][entity_types[i]]
                if add_description:
                    ax.text(j, i, f"{number_of_mistakes:.2f}\nAct: {entity_types[i]}\nPred: {entity_types[j]}",
                                ha="center", va="center", color="w")
                else:
                    ax.text(j, i, f"{number_of_mistakes:.2f}", ha="center", va="center", color="w")             

    ax.set_ylabel('Actual', fontdict=dict(weight='bold'))
    ax.set_xlabel('Predicted', fontdict=dict(weight='bold'))
    ax.set_title('Mistaken recognitions confusion matrix')

    fig.tight_layout()
    plt.show()
    
    
def plot_confusion_matrix_from_dataframe(
    dataframe: pd.DataFrame,
    target_col_name: str = 'target',
    extracted_col_name: str = 'extracted',
    id_col_name: Optional[str] = 'id',
    entity_types: Optional[list[str]] = None,
    in_percent: bool = False,
    add_description: bool = True
):
    errors = aggregate_errors_from_dataframe(dataframe, target_col_name, extracted_col_name, id_col_name)
    plot_confusion_matrix(errors, entity_types, in_percent, add_description)