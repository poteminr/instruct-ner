from razdel import tokenize
from corus.sources.rudrec import RuDReCRecord
import pandas as pd


def extract_labels_rudrec(item: RuDReCRecord) -> dict[str, list[str]] | None:
    raw_tokens = list(tokenize(item.text))
    words = [token.text for token in raw_tokens]

    word_labels = ['O'] * len(raw_tokens)
    char2word = [None] * len(item.text)

    for i, word in enumerate(raw_tokens):
        char2word[word.start:word.stop] = [i] * len(word.text)

    entity_types = ['ADR', 'DI']
    for e in item.entities:
        if e.entity_type in entity_types:
            e_words = sorted({idx for idx in char2word[e.start:e.end] if idx is not None})
            word_labels[e_words[0]] = 'B-' + e.entity_type
            for idx in e_words[1:]:
                word_labels[idx] = 'I-' + e.entity_type

    if len(set(word_labels)) != 1:
        return {'tokens': words, 'tags': word_labels}
    else:
        return None


def extract_labels_caws(
        text: str, spans: list[list[int]] | list[dict],
        data_type: str = 'hand_labeled'
) -> dict[str, list[str]]:

    raw_tokens = list(tokenize(text))
    words = [tok.text for tok in raw_tokens]

    word_labels = ['O'] * len(raw_tokens)
    char2word = [None] * len(text)

    for i, word in enumerate(raw_tokens):
        char2word[word.start:word.stop] = [i] * len(word.text)

    entity_type = "DI"
    for e in spans:
        if data_type == 'hand_labeled':
            e = e['value']
            span_start = e['start']
            span_end = e['end']
        else:
            span_start = e[0]
            span_end = e[1]

        e_words = sorted({idx for idx in char2word[span_start:span_end] if idx is not None})

        word_labels[e_words[0]] = 'B-' + entity_type
        for idx in e_words[1:]:
            word_labels[idx] = 'I-' + entity_type

    return {'tokens': words, 'tags': word_labels}


def process_rudrec(data: list[RuDReCRecord]) -> list[dict[str, list[str]]]:
    processed_data = []
    for item in data:
        extracted_data = extract_labels_rudrec(item)
        if extracted_data:
            processed_data.append(extracted_data)

    return processed_data


def process_caws(
        caws_df: pd.DataFrame,
        caws_labels: pd.DataFrame,
        hand_labeled_caws_df: pd.DataFrame
) -> list[dict[str, list[str]]]:
    processed_data = []
    for i, row in caws_df.iloc[:30].iterrows():
        extracted_data = extract_labels_caws(row['text'],
                                             caws_labels['span'][i],
                                             data_type='caws')
        processed_data.append(extracted_data)

    for _, row in hand_labeled_caws_df.iterrows():
        extracted_data = extract_labels_caws(row['data']['text'],
                                             row['annotations'][0]['result'],
                                             data_type='hand_labeled')
        processed_data.append(extracted_data)

    return processed_data
