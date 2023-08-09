import re
import numpy as np
import torch
from pybrat.parser import Example
from nested_utils.encoder_based_utils.tagset import TAGS


def text_preprocess_function(text: str) -> str:
    spaces_pattern = re.compile(r'\xa0')
    return re.sub(spaces_pattern, ' ', text)


def align_tags(example_tags: dict, tags_to_tensor: bool) -> dict:
    for label in example_tags.keys():
        example_tags[label][example_tags[label] > 1] = 1
        indexes = np.where((example_tags[label][:, 3] == 1) & (np.sum(example_tags[label], axis=1) > 1))
        example_tags[label][indexes, 3] = 0
        if tags_to_tensor:
            example_tags[label] = torch.tensor(example_tags[label])
    return example_tags


def parse_example(example: Example, tokenizer, tags_to_tensor: bool = True):
    identity_matrix = np.identity(5, dtype=int)
    B_tag = identity_matrix[0]
    I_tag = identity_matrix[1]
    L_tag = identity_matrix[2]
    O_tag = identity_matrix[3]
    U_tag = identity_matrix[4]

    tokenized_inputs = tokenizer(example.text, return_offsets_mapping=True, truncation=False)
    template = np.zeros([len(tokenized_inputs['input_ids']), 5])
    tags = dict(zip(TAGS, np.array([np.full_like(template, O_tag)] * len(TAGS))))

    for entity in example.entities:
        entity_labels = []
        flat_tags = np.full_like(template, np.zeros(5))
        span = entity.spans[0]
        label = entity.type

        start_token_idx = tokenized_inputs.char_to_token(0, span.start)
        end_token_idx = tokenized_inputs.char_to_token(0, span.end - 1)

        ids_from_span = tokenized_inputs.word_ids()[start_token_idx:end_token_idx + 1]

        for ind, word_id in enumerate(ids_from_span):
            if len(set(ids_from_span)) == 1:
                entity_labels.append(U_tag)
            elif ind == 0:
                entity_labels.append(B_tag)
            elif word_id != ids_from_span[ind - 1] and word_id != ids_from_span[-1]:
                entity_labels.append(I_tag)
            elif word_id == ids_from_span[-1]:
                entity_labels.append(L_tag)
            elif word_id == ids_from_span[ind - 1]:
                entity_labels.append(entity_labels[-1])

        flat_tags[start_token_idx:end_token_idx + 1] += entity_labels
        tags[label] += flat_tags

    tags = align_tags(tags, tags_to_tensor)
    return tags, tokenized_inputs
