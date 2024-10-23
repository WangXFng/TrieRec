from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, LlamaConfig

from transformers import Trainer

from model.collator import Collator

import argparse
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
)
from transformers import TrainingArguments

import os
os.environ["WANDB_MODE"] = "disabled"
# from fastchat.train.llama2_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )
#
# replace_llama_attn_with_flash_attn()

from model.utils import *


from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    # 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

    # args.base_model = "meta-llama/Llama-3.2-8B-Instruct"
    # args.base_model = "theo77186/Llama-3.2-8B-Instruct"
    args.base_model = "meta-llama/Llama-2-7b-hf"
    args.dataset = 'Instruments'
    args.data_path = "../data"
    args.tasks = 'seqrec'
    args.output_dir = './ckpt/Instruments-8bit-8B-4Epoch/'
    args.index_file = '.index.json'
    args.train_prompt_sample_num = '1'
    args.train_data_sample_num = '0'
    args.epochs = 4
    # args.wandb_run_name = 'test'
    args.learning_rate = 1e-4
    args.temperature = 1.0
    args.per_device_batch_size = 16

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # bnb_8bit_quant_type="nf8",
        # bnb_8bit_compute_dtype=torch.bfloat16
        # load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto")

    tokenizer = LlamaTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="left",
    )

    # vocab_size = model.get_input_embeddings().num_embeddings
    # print(f"Vocab size: {vocab_size}")

    tokenizer.pad_token_id = 0

    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config = LlamaConfig.from_pretrained(args.base_model)
    config.vocab_size = len(tokenizer)

    model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=args.lora_dropout,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        # use_dora=True, # optional DoRA
        # init_lora_weights="gaussian",
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # print(tokenizer.encode('<a_128><b_241><c_146><d_235>'))


    trainArgs = TrainingArguments(
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=16,
        warmup_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=250,
        save_strategy="no",
        optim="adamw_hf",
        push_to_hub=True,
        save_total_limit=1,
        bf16=True,
        output_dir=args.output_dir,
        dataloader_pin_memory=False,
    )
    model.config.use_cache = False

    collator = Collator(args, tokenizer)

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collator,
        args=trainArgs,
        tokenizer=tokenizer,
    )
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


    vocab_size = trainer.model.get_input_embeddings().num_embeddings
    print(f"Vocab size: {vocab_size}")