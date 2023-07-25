import pandas as pd


def calculate_metrics(extracted_list, target_list, labels=None):
    if labels is None:
        labels = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']
    overall_metrics = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in labels}
    
    for extracted, target in zip(extracted_list, target_list):
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
        
        results[label] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return results


def calculate_metrics_from_dataframe(dataframe: pd.DataFrame, labels=None, skip_empty=False):
    if labels is None:
        labels = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']
        
    if skip_empty:
        empty_template = {'Drugname': [], 'Drugclass': [], 'Drugform': [], 'DI': [], 'ADR': [], 'Finding': []}
        dataframe = dataframe[dataframe['target'] != empty_template]
        
    return calculate_metrics(dataframe['extracted'].values, dataframe['target'].values, labels)


if __name__ == '__main__':
    extracted_list = [{'Drugname': ['d'], 'Drugclass': [], 'Drugform': [], 'DI': ['боли'], 'ADR': [], 'Finding': []},
                    {'Drugname': ['k'], 'Drugclass': [], 'Drugform': [], 'DI': ['боли'], 'ADR': [], 'Finding': []}]

    target_list = [{'Drugname': ['d'], 'Drugclass': [], 'Drugform': [], 'DI': ['сидеть не могла', 'боли'], 'ADR': [], 'Finding': []},
                {'Drugname': [], 'Drugclass': [], 'Drugform': [], 'DI': ['сидеть не могла', 'боли'], 'ADR': [], 'Finding': []}]

    metrics = calculate_metrics(extracted_list, target_list)
    print(metrics)
