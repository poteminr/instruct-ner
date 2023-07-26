import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from corus import rudrec, load_rudrec
from sklearn.model_selection import train_test_split

from flat_utils.instruct_utils import entity_type_to_instruction, create_output_from_entities
from flat_utils.instruct_utils import ENTITY_TYPES, MODEL_INPUT_TEMPLATE, GENERAL_INSTRUCTION


def parse_entities_from_record(record: rudrec.RuDReCRecord) -> tuple[str, dict[str, list]]:
    entities = dict(zip(ENTITY_TYPES, [[] for _ in range(len(ENTITY_TYPES))]))
    for entity in record.entities:
        entities[entity.entity_type].append(entity.entity_text)

    return record.text, entities


def create_instructions_for_record(record: rudrec.RuDReCRecord, is_separate_labels: bool = False) -> list[dict[str, str]]:
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
        return {
            'instruction': GENERAL_INSTRUCTION,
            'input': text,
            'output': create_output_from_entities(entities, out_type=2),
            'source': MODEL_INPUT_TEMPLATE['prompts_input'].format(instruction=GENERAL_INSTRUCTION.strip(), inp=text.strip()),
            'raw_entities': entities,
            'id': f"{record.sentence_id}_{record.file_name}"
        }


def _fill_instructions_list(dataset: list[rudrec.RuDReCRecord], is_separate_labels: bool = False) -> list[dict[str, str]]:
    instructions = []
    for record in tqdm(dataset):
        if is_separate_labels:
            instructions = np.concatenate((instructions, create_instructions_for_record(record, is_separate_labels)))
        else:
            instructions.append(create_instructions_for_record(record, is_separate_labels))

    return instructions


def create_instruct_dataset(filepath: str, max_instances: int = -1, is_separate_labels: bool = False) -> list[dict[str, str]]:
    rudrec_dataset = list(load_rudrec(filepath))
    
    if max_instances != -1 and len(rudrec_dataset) > max_instances:
        rudrec_dataset = rudrec_dataset[:max_instances]

    return _fill_instructions_list(rudrec_dataset, is_separate_labels)


def create_train_test_instruct_datasets(
    filepath: str,
    max_instances: int = -1,
    is_separate_labels: bool = False,
    test_size: float = 0.3,
    random_seed: int = 42
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rudrec_dataset = list(load_rudrec(filepath))
    
    if max_instances != -1 and len(rudrec_dataset) > max_instances:
        rudrec_dataset = rudrec_dataset[:max_instances]

    train_dataset, test_dataset = train_test_split(rudrec_dataset, test_size=test_size, random_state=random_seed)
    return _fill_instructions_list(train_dataset, is_separate_labels), _fill_instructions_list(test_dataset,
                                                                                               is_separate_labels)


class InstructDataset(Dataset):
    def __init__(
        self,
        instructions: list[dict[str, str]],
        tokenizer,
        max_source_tokens_count: int,
        max_target_tokens_count: int,
        model_type: str = 'llama',
        only_target_loss: bool = True,
        padding: bool = False
    ):
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.model_type = model_type
        self.only_target_loss = only_target_loss
        self.padding = padding

        self.processed_instructions = []

        for instruction in tqdm(self.instructions):
            if self.model_type == 'llama':
                tensors = self.convert_instruction_causal(instruction)
            elif self.model_type == 't5':
                tensors = self.convert_instruction_seq2seq(instruction)
            else:
                raise ValueError('model_type must be equals "llama" or "t5"')

            self.processed_instructions.append(tensors)

    def __len__(self):
        return len(self.processed_instructions)

    def __getitem__(self, index):
        return self.processed_instructions[index]

    def convert_instruction_causal(self, instruction: dict[str, str]):
        target = instruction['output']
        source = instruction['source']        
        
        source_tokens = self.tokenizer(
            source,
            add_special_tokens=False,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True
        )['input_ids']

        if self.tokenizer.bos_token_id:
            source_tokens.insert(0, self.tokenizer.bos_token_id)

        input_ids = source_tokens[:]
        max_length = self.max_source_tokens_count + self.max_target_tokens_count + 2

        target_tokens = self.tokenizer(
            target,
            add_special_tokens=False,
            max_length=self.max_target_tokens_count,
            padding=False,
            truncation=True
        )['input_ids']

        input_ids += target_tokens + [self.tokenizer.eos_token_id]

        if self.padding:
            actual_length = len(input_ids)
            padding = [self.tokenizer.pad_token_id for i in range(len(input_ids), max_length)]
            input_ids.extend(padding)

        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())

        if self.padding:
            labels[actual_length:] = -100
            attention_mask[actual_length:] = 0

        if self.only_target_loss:
            labels[:len(source_tokens)] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    def convert_instruction_seq2seq(self, instruction: dict[str, str]):
        target = instruction['output']
        source = instruction['source']
        
        inputs = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        outputs = self.tokenizer(
            target,
            add_special_tokens=True,
            max_length=self.max_target_tokens_count,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        labels = outputs["input_ids"].squeeze(0).tolist()
        if labels[-1] != self.tokenizer.eos_token_id:
            labels.append(self.tokenizer.eos_token_id)

        inputs["labels"] = torch.LongTensor(labels)
        return inputs
