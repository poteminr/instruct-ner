from collections import defaultdict
from typing import Optional
from tqdm import tqdm
from pybrat.parser import BratParser, Example, Entity
from sklearn.model_selection import train_test_split
import numpy as np
from razdel import sentenize

from utils.instruct_dataset import Instruction
from utils.nerel_bio.nerel_bio_utils import INSTRUCTION_TEXT, ENTITY_TYPES
from utils.instruct_utils import MODEL_INPUT_TEMPLATE, create_output_from_entities


def parse_examples(data_path: str) -> list[Example]:
    parser = BratParser(ignore_types=['R'], error="ignore")
    return parser.parse(data_path)


def split_example(example: Example, n_splits: Optional[int] = None) -> tuple[str, dict[int, list[Entity]]]:
    sentences = list(sentenize(example.text))
    entities = sorted(example.entities, key=lambda x: x.spans[0].start)
    
    if n_splits is not None and n_splits > 0:
        n_splits =  min(n_splits, len(sentences))
    elif n_splits == -1:
        n_splits = len(sentences)
    else:
        n_splits = 1
        
    sentences = np.array_split(np.array(sentences), n_splits)
        
    boundings = [(a[0].start, a[-1].stop) for a in sentences]
    texts = [example.text[b[0]:b[1]] for b in boundings]
    splited_entities = defaultdict(list)
    
    for entity in entities:
        for bound_index, bound in enumerate(boundings):
            if entity.spans[0].start >= bound[0] and entity.spans[0].end <= bound[1]:
                splited_entities[bound_index].append(entity)
                break
                
    return texts, splited_entities


def parse_entities(entities: list[Entity], short_form_output: bool = True):
    if short_form_output:
        parsed_entities = defaultdict(list)
    else:
        parsed_entities = dict(zip(ENTITY_TYPES, [[] for _ in range(len(ENTITY_TYPES))]))

    for entity in entities:
        parsed_entities[entity.type].append(entity.mention)
    
    return parsed_entities

def create_instructions_for_example(
    example: Example,
    text_n_splits: Optional[int] = None,
    short_form_output: bool = True
) -> list[Instruction]:
    instructions = []
    texts, splited_entities = split_example(example, text_n_splits)
    
    for ind, text in enumerate(texts):
        entities = parse_entities(splited_entities[ind], short_form_output)
        instruction = {
            'instruction': INSTRUCTION_TEXT,
            'input': text,
            'output': create_output_from_entities(entities, out_type=2),
            'source': MODEL_INPUT_TEMPLATE['prompts_input'].format(instruction=INSTRUCTION_TEXT.strip(), inp=text.strip()),
            'raw_entities': entities,
            'id': f"{example.id}_{ind}"
        }
        instructions.append(instruction)
    
    return instructions


def _fill_instructions_list(
    examples: list[Example],
    text_n_splits: Optional[int] = None,
    short_form_output: bool = True
) -> list[Instruction]:
    instructions = []
    for example in tqdm(examples):
            instructions.extend(create_instructions_for_example(example, text_n_splits, short_form_output))

    return instructions

def create_instruct_dataset(
    data_path: str,
    max_instances: int = -1,
    text_n_splits: Optional[int] = None,
    short_form_output: bool = True
) -> list[Instruction]:
    examples = parse_examples(data_path)
    instructions = _fill_instructions_list(examples, text_n_splits, short_form_output)
    
    if max_instances != -1 and len(instructions) > max_instances:
        instructions = instructions[:max_instances]

    return instructions


def create_train_test_instruct_datasets(
    data_path: str,
    max_instances: int = -1,
    text_n_splits: Optional[int] = None,
    short_form_output: bool = True,
    test_size: float = 0.3,
    random_seed: int = 42
) -> tuple[list[Instruction], list[Instruction]]:
    examples = parse_examples(data_path)
    
    if max_instances != -1 and len(examples) > max_instances:
        examples = examples[:max_instances]

    train_dataset, test_dataset = train_test_split(examples, test_size=test_size, random_state=random_seed)
    return _fill_instructions_list(train_dataset, text_n_splits, short_form_output), \
           _fill_instructions_list(test_dataset, text_n_splits, short_form_output)    