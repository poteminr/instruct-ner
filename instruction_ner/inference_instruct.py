import re
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, GenerationConfig
from peft import PeftConfig, PeftModel
from tqdm import tqdm
import pandas as pd
from flat_utils.instruct_dataset import create_train_test_instruct_datasets
from flat_utils.instruct_utils import MODEL_INPUT_TEMPLATE
# from metric import calculate_metrics

generation_config = {
    # "bos_token_id": 1,
    "do_sample": True,
    # "eos_token_id": 2,
    "max_new_tokens": 512,
    "no_repeat_ngram_size": 20,
    "num_beams": 3,
    "pad_token_id": 0,
    "repetition_penalty": 1.1,
    "temperature": 0.9,
    "top_k": 30,
    "top_p": 0.85,
}


def extract_classes(input_string):
    answer_start_idx = input_string.find('Ответ')
    input_string = input_string[answer_start_idx+8:]
    classes = {
        "Drugname": [],
        "Drugclass": [],
        "Drugform": [],
        "DI": [],
        "ADR": [],
        "Finding": []
    }

    pattern = r"(Drugname|Drugclass|Drugform|DI|ADR|Finding):\s(.*?)(?=\n\w+:\s|$)"
    matches = re.findall(pattern, input_string)

    for class_name, value in matches:
        values = value.strip().split(', ')
        if values != ['']:
            classes[class_name] = values

    return classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rudrec_path", default='data/rudrec/rudrec_annotated.json', type=str, help='train file_path')
    parser.add_argument("--model_type", default='llama', type=str, help='model type')
    parser.add_argument("--model_name", default='poteminr/llama2-rudrec', type=str, help='model name from hf')
    parser.add_argument("--max_instances", default=-1, type=int, help='max number of instruction')
    parser.add_argument("--callback", default=False, type=bool, help='print predictions')

    
    arguments = parser.parse_args()

    model_name = arguments.model_name
    
    peft_config = PeftConfig.from_pretrained(arguments.model_name)
    base_model_name = peft_config.base_model_name_or_path
    
    models = {'llama': AutoModelForCausalLM, 't5': T5ForConditionalGeneration}
    
    model = models[arguments.model_type].from_pretrained(
        base_model_name,
        load_in_8bit=True,
        device_map='auto'
    )
    
    model = PeftModel.from_pretrained(model, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    
    _, test_dataset = create_train_test_instruct_datasets(arguments.rudrec_path)
    if arguments.max_instances != -1 and arguments.max_instances < len(test_dataset):
        test_dataset = test_dataset[:arguments.max_instances]
    
    extracted_list = []
    target_list = []
    instruction_ids = []
    for instruction in tqdm(test_dataset):
        # instruction = test_dataset[arguments.rudrec_index]
        inst = instruction['instruction']
        inp = instruction['input']
        target = instruction['output'].strip()
        raw_entities = instruction['raw_entities']

        source = MODEL_INPUT_TEMPLATE['prompts_input'].format(instruction=inst.strip(), inp=inp.strip())
        input_ids = tokenizer(source, return_tensors="pt")["input_ids"].cuda()
        
        # print("Generating...")
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(**generation_config),
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )
        for s in generation_output.sequences:
            string_output = tokenizer.decode(s, skip_special_tokens=True)
            extracted_entities = extract_classes(string_output)

            if arguments.callback:
                print(string_output)
                print("TARGET:")
                print(target)
                print("Extracted:")
                print(extracted_entities)
                print("Extracted Target")
                print(raw_entities)
            
            extracted_list.append([extracted_entities])
            target_list.append([raw_entities])
            instruction_ids.append(instruction['id'])
    
    pd.DataFrame({
        'id': instruction_ids, 
        'extracted': extracted_list,
        'target': target_list
    }).to_json('prediction.json')