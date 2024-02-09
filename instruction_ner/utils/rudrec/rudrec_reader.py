import numpy as np
from tqdm import tqdm
from corus import rudrec, load_rudrec
from sklearn.model_selection import train_test_split
from typing import Union

from utils.instruct_dataset import Instruction
from instruction_ner.utils.rudrec.rudrec_utils import ENTITY_TYPES, INSTRUCTION_TEXT, EXTENDED_INSTRUCTION_TEXT, entity_type_to_instruction
from utils.instruct_utils import MODEL_INPUT_TEMPLATE, create_output_from_entities


def parse_entities_from_record(record: rudrec.RuDReCRecord) -> tuple[str, dict[str, list]]:
    entities = dict(zip(ENTITY_TYPES, [[] for _ in range(len(ENTITY_TYPES))]))
    for entity in record.entities:
        entities[entity.entity_type].append(entity.entity_text)

    return record.text, entities


def create_instructions_for_record(
    record: rudrec.RuDReCRecord,
    is_separate_labels: bool = False,
    extended_instruction_text: bool = True
) -> Union[list[Instruction], Instruction]:
    text, entities = parse_entities_from_record(record)
    if is_separate_labels:
        record_instructions = []
        for entity_type in entities.keys():
            instruction = entity_type_to_instruction(entity_type)
            record_instructions.append({
                'instruction': instruction,
                'input': text,
                'output': create_output_from_entities(entities[entity_type]),
                'source': MODEL_INPUT_TEMPLATE['prompts_input'].format(instruction=instruction.strip(), inp=text.strip()),
                'label': entity_type,
                'id': f"{record.sentence_id}_{record.file_name}"
            })
        return record_instructions
    else:
        instruction_text = EXTENDED_INSTRUCTION_TEXT if extended_instruction_text else INSTRUCTION_TEXT
        return {
            'instruction': instruction_text,
            'input': text,
            'output': create_output_from_entities(entities, out_type=2),
            'source': MODEL_INPUT_TEMPLATE['prompts_input'].format(instruction=instruction_text.strip(), inp=text.strip()),
            'raw_entities': entities,
            'id': f"{record.sentence_id}_{record.file_name}"
        }


def _fill_instructions_list(dataset: list[rudrec.RuDReCRecord], is_separate_labels: bool = False) -> list[Instruction]:
    instructions = []
    for record in tqdm(dataset):
        if is_separate_labels:
            instructions = np.concatenate((instructions, create_instructions_for_record(record, is_separate_labels)))
        else:
            instructions.append(create_instructions_for_record(record, is_separate_labels))

    return instructions


def create_instruct_dataset(data_path: str, max_instances: int = -1, is_separate_labels: bool = False) -> list[Instruction]:
    rudrec_dataset = list(load_rudrec(data_path))
    
    if max_instances != -1 and len(rudrec_dataset) > max_instances:
        rudrec_dataset = rudrec_dataset[:max_instances]

    return _fill_instructions_list(rudrec_dataset, is_separate_labels)


def create_train_test_instruct_datasets(
    data_path: str,
    max_instances: int = -1,
    is_separate_labels: bool = False,
    test_size: float = 0.3,
    random_seed: int = 42
) -> tuple[list[Instruction], list[Instruction]]:
    rudrec_dataset = list(load_rudrec(data_path))
    
    if max_instances != -1 and len(rudrec_dataset) > max_instances:
        rudrec_dataset = rudrec_dataset[:max_instances]

    train_dataset, test_dataset = train_test_split(rudrec_dataset, test_size=test_size, random_state=random_seed)
    return _fill_instructions_list(train_dataset, is_separate_labels), _fill_instructions_list(test_dataset,
                                                                                               is_separate_labels)
    