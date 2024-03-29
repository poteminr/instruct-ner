from tqdm import tqdm
from datasets import Dataset, load_dataset
from collections import defaultdict 
from utils.instruct_dataset import Instruction
from utils.multiconer2023.multiconer_utils import ENTITY_TYPES, INSTRUCTION_TEXT, COARSE_INSTRUCTION_TEXT, preprocess_entity_type
from utils.instruct_utils import MODEL_INPUT_TEMPLATE, create_output_from_entities


def parse_entities_from_sample(
    ner_tags: list[int],
    tokens: list[str],
    short_form_output: bool = True,
    coarse_level_tagset: bool = False
) -> dict[str, list[str]]:
    if short_form_output:
        entities = defaultdict(list)
    else:
        entities = dict(zip(ENTITY_TYPES, [[] for _ in range(len(ENTITY_TYPES))]))
    current_category = None  # Initialize the current category
    current_entity = []  # Initialize the current entity

    # Iterate through the tags and tokens
    for tag_label, token in zip(ner_tags, tokens):
        if tag_label == 'O':
            if current_entity:  # Check if there is a current entity
                entities[preprocess_entity_type(current_category, coarse_level_tagset)].append(" ".join(current_entity))
                current_entity = []  # Reset the current entity
            current_category = None  # Reset the current category
        else:
            category = tag_label.split('-')[1]  # Extract the category (e.g., 'PER' from 'B-PER')
                
            if tag_label.startswith('B-'):
                if current_entity:  # Check if there is a current entity
                    entities[preprocess_entity_type(current_category, coarse_level_tagset)].append(" ".join(current_entity))
                    current_entity = []  # Reset the current entity
                current_category = category
                current_entity.append(token)
            elif tag_label.startswith('I-') and current_category is not None:
                current_entity.append(token)

    # Check if there is a remaining entity
    if current_entity:
        entities[preprocess_entity_type(current_category, coarse_level_tagset)].append(" ".join(current_entity))
    
    return entities


def create_instructions_for_sample(
    sample: dict[str, list],
    short_form_output: bool = True,
    coarse_level_tagset: bool = False
) -> Instruction:
    text = " ".join(sample['tokens'])
    entities = parse_entities_from_sample(sample['ner_tags'], sample['tokens'], short_form_output, coarse_level_tagset)
    instruction_text = COARSE_INSTRUCTION_TEXT if coarse_level_tagset else INSTRUCTION_TEXT 
    return {
        'instruction': instruction_text,
        'input': text,
        'output': create_output_from_entities(entities, out_type=2),
        'source': MODEL_INPUT_TEMPLATE['prompts_input'].format(instruction=instruction_text.strip(), inp=text.strip()),
        'raw_entities': entities,
        'id': f"{sample['id']}"
    }
    
    
def _fill_instructions_list(
    dataset: Dataset,
    short_form_output: bool = True,
    coarse_level_tagset: bool = False
) -> list[Instruction]:
    instructions = []
    for sample in tqdm(dataset):
        instructions.append(create_instructions_for_sample(sample, short_form_output, coarse_level_tagset))
        
    return instructions

def create_instruct_dataset(
    split: str,
    shuffle: bool = False,
    max_instances: int = -1,
    short_form_output: bool = True,
    coarse_level_tagset: bool = False    
) -> list[Instruction]:
    dataset = load_dataset('MultiCoNER/multiconer_v2', 'English (EN)', split=split)    
    if shuffle:
        dataset = dataset.shuffle(seed=42)
        
    instructions = _fill_instructions_list(dataset, short_form_output, coarse_level_tagset)
    
    if max_instances != -1 and len(instructions) > max_instances:
        instructions = instructions[:max_instances]

    return instructions
