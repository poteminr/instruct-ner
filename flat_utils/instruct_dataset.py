import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from corus import rudrec, load_rudrec
from sklearn.model_selection import train_test_split
from flat_utils.instruct_utils import SEP_SYMBOL, ENTITY_TYPES, MODEL_INPUT_TEMPLATE, entity_type_to_instruction


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


def create_instruct_dataset(filepath: str) -> list[dict[str, str]]:
    instructions = []
    rudrec_dataset = list(load_rudrec(filepath))
    for record in tqdm(rudrec_dataset):
        instructions = np.concatenate((instructions, create_instructions_for_record(record)))
    
    return instructions


def create_train_test_instruct_datasets(
        filepath: str,
        test_size: float = 0.3,
        random_seed: int = 42
) -> tuple[list[dict[str, str]]]:
    
    train_instructions, test_instructions = [], []
    rudrec_dataset = list(load_rudrec(filepath))
    train_dataset, test_dataset = train_test_split(rudrec_dataset, test_size=test_size, random_state=random_seed)

    for record in tqdm(train_dataset):
        train_instructions = np.concatenate((train_instructions, create_instructions_for_record(record)))
        
    for record in tqdm(test_dataset):
        test_instructions = np.concatenate((test_instructions, create_instructions_for_record(record)))
    
    return train_instructions, test_instructions


class InstructDataset(Dataset):
    def __init__(self, instructions: list[dict[str, str]], tokenizer, only_target_loss: bool = True):
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.only_target_loss = only_target_loss
        self.template = MODEL_INPUT_TEMPLATE
        self.processed_instructions = []
        
        for instruction in tqdm(self.instructions):
            tensors = self.convert_instruction(instruction)
            self.processed_instructions.append(tensors)
            
    def __len__(self):
        return len(self.processed_instructions)
    
    def __getitem__(self, index):
        return self.processed_instructions[index]
        
    def convert_instruction(self, instruction: dict):
        inst = instruction['instruction']
        inp = instruction['input']
        target = instruction['output'].strip()
        
        source = self.template['prompts_input'].format(instruction=inst.strip(), inp=inp.strip())
        source_tokens = self.tokenizer(source, add_special_tokens=False, padding=False)['input_ids']
        
        if self.tokenizer.bos_token_id:
            source_tokens.insert(0, self.tokenizer.bos_token_id)
            
        input_ids = source_tokens[:]
        
        target_tokens = self.tokenizer(target, add_special_tokens=False, padding=False)['input_ids']
        input_ids += target_tokens + [self.tokenizer.eos_token_id]
        
        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())
        
        if self.only_target_loss:
            labels[:len(source_tokens)] = -100
            
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
        