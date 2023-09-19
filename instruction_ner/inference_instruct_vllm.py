import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig
from vllm import LLM, SamplingParams
from utils.rudrec.rudrec_reader import create_train_test_instruct_datasets
from utils.nerel_bio.nerel_reader import create_instruct_dataset
from metric import extract_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='rudrec', type=str, help='name of dataset')
    parser.add_argument("--data_path", default='data/rudrec/rudrec_annotated.json', type=str, help='train file_path')
    parser.add_argument("--model_type", default='llama', type=str, help='model type')
    parser.add_argument("--model_name", default='poteminr/llama2-rudrec-merged', type=str, help='model name from hf')
    parser.add_argument("--prediction_path", default='prediction.json', type=str, help='path for saving prediction')
    parser.add_argument("--max_instances", default=-1, type=int, help='max number of instruction')
    arguments = parser.parse_args()

    try:
        model = LLM(model=arguments.model_name)
    except OSError as e:
        raise type(e)(str(e) +' You should merge your adapter with base model.').with_traceback(sys.exc_info()[2])
    
    generation_config = GenerationConfig.from_pretrained(arguments.model_name)
    
    
    sampling_params = SamplingParams(
        best_of=generation_config.num_beams,
        temperature=0,
        top_k=-1,
        top_p=1,
        use_beam_search=(generation_config.num_beams > 1),
        length_penalty=generation_config.length_penalty,
        early_stopping=generation_config.early_stopping,
        max_tokens=generation_config.max_length
    )
    
    if arguments.dataset_name == 'rudrec': 
        from utils.rudrec.rudrec_utis import ENTITY_TYPES
        _, test_dataset = create_train_test_instruct_datasets(arguments.data_path)
        if arguments.max_instances != -1 and arguments.max_instances < len(test_dataset):
            test_dataset = test_dataset[:arguments.max_instances]
    else:
        from utils.nerel_bio.nerel_bio_utils import ENTITY_TYPES
        test_path = os.path.join(arguments.data_path, 'test')
        test_dataset = create_instruct_dataset(test_path, max_instances=arguments.max_instances)
    
    extracted_list = []
    target_list = []
    instruction_ids = []
    sources = []
    for instruction in tqdm(test_dataset):
        target_list.append(instruction['raw_entities'])
        instruction_ids.append(instruction['id'])
        sources.append(instruction['source'])

    outputs = model.generate(sources, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        extracted_list.append(extract_classes(generated_text, ENTITY_TYPES))
        
    pd.DataFrame({
        'id': instruction_ids, 
        'extracted': extracted_list,
        'target': target_list
    }).to_json(arguments.prediction_path)