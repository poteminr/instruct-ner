import numpy as np
from tqdm import tqdm
from corus import rudrec, load_rudrec
from sklearn.model_selection import train_test_split
from flat_utils.instruct_utils import SEP_SYMBOL, ENTITY_TYPES, entity_type_to_instruction


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
        output = SEP_SYMBOL.join(entities[entity_type])
        record_instructions.append({
            'instruction': instruction,
            'input': text,
            'output': output,
            'label': entity_type,
            'id': f"{record.sentence_id}_{record.file_name}"
        })
    return record_instructions


def create_instruct_dataset(filepath: str) -> np.ndarray[dict[str, str]]:
    instructions = []
    rudrec_dataset = download_dataset(filepath)
    for record in tqdm(rudrec_dataset):
        instructions = np.concatenate((instructions, create_instructions_for_record(record)))
    
    return instructions


def create_train_test_instruct_datasets(
        filepath: str,
        test_size: float = 0.3,
        random_seed: int = 42
) -> tuple[np.ndarray[dict[str, str]]]:
    
    train_instructions, test_instructions = [], []
    rudrec_dataset = download_dataset(filepath)
    train_dataset, test_dataset = train_test_split(rudrec_dataset, test_size=test_size, random_state=random_seed)

    for record in tqdm(train_dataset):
        train_instructions = np.concatenate((train_instructions, create_instructions_for_record(record)))
        
    for record in tqdm(test_dataset):
        test_instructions = np.concatenate((test_instructions, create_instructions_for_record(record)))
    
    return train_instructions, test_instructions