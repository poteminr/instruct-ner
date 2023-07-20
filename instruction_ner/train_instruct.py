import json
import os
import wandb
import torch
import argparse
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl, GPT2Tokenizer, T5ForConditionalGeneration
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, PeftConfig, PeftModel

from flat_utils.instruct_dataset import InstructDataset, create_train_test_instruct_datasets
from train_utils import fix_tokenizer, fix_model, set_random_seed


# https://github.com/huggingface/peft/issues/96
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control
     

def train(
    train_instructions: list[dict[str, str]],
    test_instructions: list[dict[str, str]],
    model_type: str,
    output_dir: str,
    seed: int,
    config_file: str
):
    set_random_seed(seed)
    with open(config_file, "r") as r:
        config = json.load(r)

    lora_config = config.get("lora")

    model_name = config['model_name']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer)

    only_target_loss = config.get("only_target_loss", True)
    
    max_source_tokens_count = config["max_source_tokens_count"]
    max_target_tokens_count = config["max_target_tokens_count"]

    train_dataset = InstructDataset(
        train_instructions,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        model_type=model_type,
        only_target_loss=only_target_loss
    )

    val_dataset = InstructDataset(
        test_instructions,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        model_type=model_type,
        only_target_loss=only_target_loss
    )   
    
    model_classes = {
        'llama': {
            'data_collator': DataCollatorForTokenClassification,
            'model': AutoModelForCausalLM
        },
        't5': {
            'data_collator': DataCollatorForSeq2Seq,
            'model': T5ForConditionalGeneration
        }
    }
    
    data_collator = model_classes[model_type]['data_collator'](tokenizer, pad_to_multiple_of=8)

    load_in_8bit = bool(config.get("load_in_8bit", True))
    is_adapter = config['is_adapter']

    if load_in_8bit:
        if is_adapter:
            peft_config = PeftConfig.from_pretrained(model_name)
            model = model_classes[model_type]['model'].from_pretrained(
                peft_config.base_model_name_or_path,
                load_in_8bit=True,
                device_map='auto'
            )
            model = fix_model(model, tokenizer, use_resize=False)
            model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(model, model_name, is_trainable=True)
        else:
            model = model_classes[model_type]['model'].from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map='auto'
            )
            model = fix_model(model, tokenizer, use_resize=False)
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(**lora_config) 
            model = get_peft_model(model, peft_config)            
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = fix_model(model, tokenizer)

    # Default model generation params
    model.config.num_beams = 5
    max_tokens_count = max_target_tokens_count + max_source_tokens_count + 1
    model.config.max_length = max_tokens_count

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    deepspeed_config = config.get("deepspeed")
    trainer_config = config["trainer"]
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to='wandb',
        ddp_find_unused_parameters=None,
        deepspeed=deepspeed_config,
        **trainer_config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[SavePeftModelCallback],
        data_collator=data_collator
    )

    with wandb.init(project="Instruction NER") as run:
        model.print_trainable_parameters()
        trainer.train()
        if 'llama2' in config_file:
            model_type = 'llama2'
        model.push_to_hub(f"poteminr/{model_type}-rudrec", use_auth_token=True)
        tokenizer.push_to_hub(f"poteminr/{model_type}-rudrec", use_auth_token=True)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rudrec_path", default='data/rudrec/rudrec_annotated.json', type=str, help='train file_path')
    parser.add_argument("--output_dir", default='models/llama-rudrec', type=str, help='output_dir')
    parser.add_argument("--test_size", default=0.3, type=float, help='test_size')
    parser.add_argument("--random_seed", default=42, type=int, help='random_seed')
    parser.add_argument("--config_file", default='configs/llama_7b_lora.json', type=str, help='path to config file')
    parser.add_argument("--model_type", default='llama', type=str, help='model type')
    parser.add_argument("--max_instances", default=-1, type=int, help='max number of rudrec records')

    arguments = parser.parse_args()
    
    train_dataset, test_dataset = create_train_test_instruct_datasets(
        filepath=arguments.rudrec_path,
        max_instances=arguments.max_instances,
        test_size=arguments.test_size,
        random_seed=arguments.random_seed
    )
    
    train(
        train_instructions=train_dataset,
        test_instructions=test_dataset,
        model_type=arguments.model_type,
        output_dir=arguments.output_dir,
        seed=arguments.random_seed,
        config_file=arguments.config_file
    )
    