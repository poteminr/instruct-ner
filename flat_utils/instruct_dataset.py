import numpy as np
from tqdm import tqdm
from corus import rudrec, load_rudrec
from flat_utils.instruct_utils import ENTITY_TYPES, entity_type_to_instruction


def download_dataset(filepath: str) -> list:
    return list(load_rudrec(filepath))


def parse_entities_from_record(record: rudrec.RuDReCRecord) -> tuple[str, dict[str, list]]:
    entities = dict(zip(ENTITY_TYPES, [[] for _ in range(len(ENTITY_TYPES))]))
    for entity in record.entities:
        entities[entity.entity_type].append(entity.entity_text)
    
    return record.text, entities


def create_instructions_for_record(record: rudrec.RuDReCRecord) -> list[dict[str, str]]:
    record_instructions = []
    text, entities = parse_entities_from_record(record)
    for entity_type in entities.keys():
        instruction = entity_type_to_instruction(entity_type)
        output = ' <s> '.join(entities[entity_type])
        record_instructions.append({
            'instruction': instruction,
            'input': text,
            'output': output
        })
    return record_instructions


def create_instruct_dataset(filepath: str) -> list[dict[str, str]]:
    instructions = []
    rudrec_dataset = download_dataset(filepath)
    for record in tqdm(rudrec_dataset):
        instructions = np.concatenate((instructions, create_instructions_for_record(record)))
    
    return instructions
