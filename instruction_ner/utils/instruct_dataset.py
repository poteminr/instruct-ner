import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import TypedDict


class Instruction(TypedDict):
    instruction: str
    input: str
    output: str
    source: str   
    raw_entities: dict[str, list[str]]
    id: str


class InstructDataset(Dataset):
    def __init__(
        self,
        instructions: list[Instruction],
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
