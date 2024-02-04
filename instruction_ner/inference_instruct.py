import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, GenerationConfig
from peft import PeftConfig, PeftModel

from metric import extract_classes
from train_utils import SUPPORTED_DATASETS, MODEL_CLASSES


def batch(iterable, n=4):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='rudrec', type=str, help='name of dataset')
    parser.add_argument("--data_path", default='data/rudrec/rudrec_annotated.json', type=str, help='train file_path')
    parser.add_argument("--model_type", default='llama', type=str, help='model type')
    parser.add_argument("--model_name", default='poteminr/llama2-rudrec', type=str, help='model name from hf')
    parser.add_argument("--prediction_path", default='prediction.json', type=str, help='path for saving prediction')
    parser.add_argument("--max_instances", default=-1, type=int, help='max number of instruction')
    parser.add_argument("--text_n_splits", default=-1, type=int, help='number of splits for nerel')
    parser.add_argument("--coarse_tagset_multiconer", default=False, type=bool, help='use_coarse_tagset_multiconer')
    parser.add_argument("--batch_size", default=4, type=int, help='number of instructions in batch')
    arguments = parser.parse_args()

    assert arguments.dataset_name in SUPPORTED_DATASETS, f'expected dataset name from {SUPPORTED_DATASETS}'

    generation_config = GenerationConfig.from_pretrained(arguments.model_name)
    peft_config = PeftConfig.from_pretrained(arguments.model_name)
    
    model = MODEL_CLASSES[arguments.model_type]['model'].from_pretrained(
        peft_config.base_model_name_or_path,
        load_in_8bit=True,
        device_map='auto'
    )
    
    model = PeftModel.from_pretrained(model, arguments.model_name)
    tokenizer = AutoTokenizer.from_pretrained(arguments.model_name)
    
    model.eval()
    model = torch.compile(model)
    
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    if arguments.dataset_name == 'rudrec': 
        from utils.rudrec.rudrec_reader import create_train_test_instruct_datasets
        from utils.rudrec.rudrec_utis import ENTITY_TYPES

        _, test_dataset = create_train_test_instruct_datasets(arguments.data_path)
        if arguments.max_instances != -1 and arguments.max_instances < len(test_dataset):
            test_dataset = test_dataset[:arguments.max_instances]
    elif arguments.dataset_name == 'nerel_bio':
        from utils.nerel_bio.nerel_reader import create_instruct_dataset
        from utils.nerel_bio.nerel_bio_utils import ENTITY_TYPES

        test_dataset = create_instruct_dataset(
            data_path=os.path.join(arguments.data_path, 'test'),
            max_instances=arguments.max_instances,
            text_n_splits=arguments.text_n_splits
        )
    elif arguments.dataset_name == 'conll2003':
        from utils.conll2003.conll_reader import create_instruct_dataset
        from utils.conll2003.conll_utils import ENTITY_TYPES
    
        test_dataset = create_instruct_dataset(split='test', max_instances=arguments.max_instances)
    elif arguments.dataset_name == 'multiconer2023':
        from utils.multiconer2023.multiconer_reader import create_instruct_dataset
        from utils.multiconer2023.multiconer_utils import ENTITY_TYPES, COARSE_ENTITY_TYPES
        
        if arguments.coarse_tagset_multiconer:
            ENTITY_TYPES = COARSE_ENTITY_TYPES
            
        test_dataset = create_instruct_dataset(
            split='test',
            shuffle=True,
            max_instances=arguments.max_instances,
            coarse_level_tagset=arguments.coarse_tagset_multiconer
        )
            
    extracted_list = []
    target_list = []
    instruction_ids = []
    sources = []
    
    for instruction in tqdm(test_dataset):
        target_list.append(instruction['raw_entities'])
        instruction_ids.append(instruction['id'])
        sources.append(instruction['source'])
       
    target_list = list(batch(target_list, n=arguments.batch_size))
    instruction_ids = list(batch(instruction_ids, n=arguments.batch_size))    
    sources = list(batch(sources, n=arguments.batch_size))
    
    for source in tqdm(sources):
        input_ids = tokenizer(source, return_tensors="pt", padding=True)["input_ids"].cuda()
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )
        for s in generation_output.sequences:
            string_output = tokenizer.decode(s, skip_special_tokens=True)
            extracted_list.append(extract_classes(string_output, ENTITY_TYPES))
    
    pd.DataFrame({
        'id': np.concatenate(instruction_ids), 
        'extracted': extracted_list,
        'target': np.concatenate(target_list)
    }).to_json(arguments.prediction_path)