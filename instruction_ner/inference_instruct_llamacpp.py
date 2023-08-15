import re
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig
from llama_cpp import Llama
from utils.rudrec.rudrec_reader import create_train_test_instruct_datasets
from utils.rudrec.rudrec_utis import ENTITY_TYPES
from metric import extract_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rudrec_path", default='data/rudrec/rudrec_annotated.json', type=str, help='train file_path')
    parser.add_argument("--model_path", default='poteminr/llama2-rudrec', type=str, help='ggml model path')
    parser.add_argument("--model_name", default='poteminr/llama2-rudrec', type=str, help='model name from hf')
    parser.add_argument("--prediction_path", default='prediction.json', type=str, help='path for saving prediction')
    parser.add_argument("--max_instances", default=-1, type=int, help='max number of instruction')
    parser.add_argument("--max_new_tokens", default=512, type=int, help='max number of generated tokens')
    arguments = parser.parse_args()

    model = Llama(
        model_path=arguments.model_path,
        n_gpu_layers = 35,
        n_ctx=2048,
        n_parts=1,
        use_mmap=False,
    )
    generation_config = GenerationConfig.from_pretrained(arguments.model_name)
    max_new_tokens = arguments.max_new_tokens
    
    _, test_dataset = create_train_test_instruct_datasets(arguments.rudrec_path)
    if arguments.max_instances != -1 and arguments.max_instances < len(test_dataset):
        test_dataset = test_dataset[:arguments.max_instances]
    
    extracted_list = []
    target_list = []
    instruction_ids = []
    sources = []
    
    for instruction in tqdm(test_dataset):
        input_ids = model.tokenize(instruction['source'])
        input_ids.append(model.token_eos())
        generator = model.generate(
                input_ids,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                temp=generation_config.temperature,
                repeat_penalty=generation_config.repetition_penalty,
                reset=True,
        )

        completion_tokens = []
        for i, token in enumerate(generator):
            completion_tokens.append(token)
            if token == model.token_eos() or (max_new_tokens is not None and i >= max_new_tokens):
                break
            
        completion_tokens = model.detokenize(completion_tokens).decode("utf-8")
        extracted_list.append(extract_classes(completion_tokens), ENTITY_TYPES)
        instruction_ids.append(instruction['id'])
        target_list.append(instruction['raw_entities'])
    
    pd.DataFrame({
        'id': instruction_ids, 
        'extracted': extracted_list,
        'target': target_list
    }).to_json(arguments.prediction_path)