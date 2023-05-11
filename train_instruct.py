import json
import os
import wandb
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training, PeftConfig
from flat_utils.instruct_dataset import InstructDataset, create_train_test_instruct_datasets
from train_utils import fix_tokenizer, fix_model, set_random_seed



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
    train_instructions,
    test_instructions,
    output_dir: str = 'models/llama_7b_lora',
    seed: int = 42,
    config_file: str = 'llama_7b_lora.json',

):
    set_random_seed(seed)
    with open(config_file, "r") as r:
        config = json.load(r)

    device_map = "auto"


    deepspeed_config = config.get("deepspeed")
    trainer_config = config["trainer"]
    lora_config = config.get("lora")
    callbacks = [SavePeftModelCallback] if lora_config else []
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to='wandb',
        deepspeed=deepspeed_config,
        **trainer_config
    )

    model_name = 'IlyaGusev/llama_7b_ru_turbo_alpaca_lora'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = fix_tokenizer(tokenizer)
    tokenizer.save_pretrained(output_dir)

    model_type = config.get("model_type", "causal")
    templates_path = config.get("templates_path", "ru_alpaca_template.json")
    only_target_loss = config.get("only_target_loss", True)
    mode = config.get("mode", "instruct")
    if mode == "instruct":
        max_source_tokens_count = config["max_source_tokens_count"]
        max_target_tokens_count = config["max_target_tokens_count"]
        target_field = config.get("target_field", "output")
        source_field = config.get("source_field", "input")

        train_dataset = InstructDataset(
            train_instructions,
            tokenizer,
            only_target_loss=only_target_loss
        )

        val_dataset = InstructDataset(
            test_instructions,
            tokenizer,
            only_target_loss=only_target_loss
        )
        

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    print("INPUT_IDS")
    print(data_collator([train_dataset[0], train_dataset[1]])["input_ids"][0])
    print("MASK")
    print(data_collator([train_dataset[0], train_dataset[1]])["attention_mask"][0])
    print("LABELS")
    print(data_collator([train_dataset[0], train_dataset[1]])["labels"][0])


    load_in_8bit = bool(config.get("load_in_8bit", False))
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            PeftConfig.from_pretrained(model_name).base_model_name_or_path,
            load_in_8bit=True,
            device_map=device_map
        )
        model = fix_model(model, tokenizer, use_resize=False)
        model = prepare_model_for_int8_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = fix_model(model, tokenizer)

    # Default model generation params
    model.config.num_beams = 5
    if mode == "instruct":
        max_tokens_count = max_target_tokens_count + max_source_tokens_count + 1
    model.config.max_length = max_tokens_count if model_type == "causal" else max_target_tokens_count

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)

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
        callbacks=callbacks,
        data_collator=data_collator
    )

    with wandb.init(project="rulm_self_instruct", name=config_file) as run:
        trainer.train()
        model.save_pretrained(output_dir)
        
        
if __name__ == "__main__":
    train_dataset, test_dataset = create_train_test_instruct_datasets('data/rudrec/rudrec_annotated.json')
    train(train_dataset, train_dataset)