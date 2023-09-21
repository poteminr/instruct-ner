from tqdm import tqdm
import datasets
from datasets import Dataset
from typing import Union

from utils.instruct_dataset import Instruction
from utils.conll2003.conll_utils import ENTITY_TYPES, INSTRUCTION_TEXT, TAGSET

from utils.instruct_utils import MODEL_INPUT_TEMPLATE, create_output_from_entities


def parse_entities_from_sample(ner_tags: list[int], tokens: list[str]) -> dict[str, list[str]]:
    entities = dict(zip(ENTITY_TYPES, [[] for _ in range(len(ENTITY_TYPES))]))

    current_category = None  # Initialize the current category
    current_entity = []  # Initialize the current entity

    # Iterate through the tags and tokens
    for tag, token in zip(ner_tags, tokens):
        if tag == 0:
            if current_entity:  # Check if there is a current entity
                entities[current_category].append(" ".join(current_entity))
                current_entity = []  # Reset the current entity
            current_category = None  # Reset the current category
        else:
            tag_label = list(TAGSET.keys())[list(TAGSET.values()).index(tag)]  # Get the tag label
            category = tag_label.split('-')[1]  # Extract the category (e.g., 'PER' from 'B-PER')

            if tag_label.startswith('B-'):
                if current_entity:  # Check if there is a current entity
                    entities[current_category].append(" ".join(current_entity))
                    current_entity = []  # Reset the current entity
                current_category = category
                current_entity.append(token)
            elif tag_label.startswith('I-') and current_category is not None:
                current_entity.append(token)

    # Check if there is a remaining entity
    if current_entity:
        entities[current_category].append(" ".join(current_entity))
    
    return entities

def create_instructions_for_sample(sample: dict[str, list]) -> Instruction:
    text = " ".join(sample['tokens'])
    entities = parse_entities_from_sample(sample['ner_tags'], sample['tokens'])
    
    return {
        'instruction': INSTRUCTION_TEXT,
        'input': text,
        'output': create_output_from_entities(entities, out_type=2),
        'source': MODEL_INPUT_TEMPLATE['prompts_input'].format(instruction=INSTRUCTION_TEXT.strip(), inp=text.strip()),
        'raw_entities': entities,
        'id': f"{sample['id']}"
    }
    
def _fill_instructions_list(dataset: Dataset) -> list[Instruction]:
    instructions = [create_instructions_for_sample(sample) for sample in tqdm(dataset)]
    return instructions

def create_instruct_dataset(split: str, max_instances: int = -1) -> list[Instruction]:
    dataset = datasets.load_dataset('conll2003', split=split)    
    instructions = _fill_instructions_list(dataset)
    
    if max_instances != -1 and len(instructions) > max_instances:
        instructions = instructions[:max_instances]

    return instructions
