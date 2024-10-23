from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

from transformers import Trainer
from transformers import LlamaForCausalLM, LlamaTokenizer, LogitsProcessorList, LlamaConfig
from model.TrieLogistsProcesor import Trie, TrieMachine, TrieLogitsProcessor

from peft import PeftModel

from model.collator import TestCollator
from torch.utils.data import DataLoader

import argparse
from model.utils import *
from model.evaluate import get_topk_results, get_metrics_results

from tqdm import tqdm
from model.prompt import all_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()

    # args.base_model = "meta-llama/Llama-2-7b-hf"
    args.base_model = "meta-llama/Llama-3.2-1B-Instruct"
    # args.base_model = "meta-llama/Llama-3.2-3B-Instruct"
    # args.base_model = "huggyllama/llama-7b"
    # args.ckpt_path = './ckpt/Instruments-8bit-3B-4Epoch-EmbToken/'
    # args.ckpt_path = './ckpt/Instruments-8bit-1B-4Epoch-Incre/'
    args.ckpt_path = './ckpt/Instruments-8bit-1B-4Epoch/'
    args.dataset = 'Instruments'
    args.data_path = "./data"
    args.tasks = 'seqrec'
    args.test_task = 'seqrec'
    args.index_file = '.index.json'
    # args.train_prompt_sample_num = '1'
    # args.train_data_sample_num = '1'
    args.wandb_run_name = 'test'
    # args.learning_rate = 1e-4
    args.temperature = 1.0
    args.per_device_batch_size = 1
    args.test_batch_size = 16
    args.num_beams = 20

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto")

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        model,
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # vocab_size = model.get_input_embeddings().num_embeddings
    # print(f"Vocab size: {vocab_size}")
    model.print_trainable_parameters()

    test_data = load_test_dataset(args)
    all_items = test_data.get_all_items()

    ## ===================== ID Pattern =====================
    encoded_sequences = []
    for sequence in all_items:
        token_ids = tokenizer.encode(sequence)
        encoded_sequences.append(token_ids[1:])

    trie = TrieMachine(tokenizer.eos_token_id, encoded_sequences).getRoot()

    # customized LogitsProcessor
    logits_processor = LogitsProcessorList([TrieLogitsProcessor(trie, tokenizer, args.num_beams, last_token=':')])
    #
    # ## ===================== ID Pattern =====================
    # prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)

    collator = TestCollator(args, tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator
                             , num_workers=2, pin_memory=True)
    device = torch.device("cuda",0)

    model.eval()
    metrics = args.metrics.split(",")
    all_prompt_results = []
    prompt_ids = range(len(all_prompt["seqrec"]))
    with torch.no_grad():
        metrics_results = {}
        total = 0
        for prompt_id in prompt_ids:
            test_loader.dataset.set_prompt(prompt_id)
            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                bs = len(targets)
                num_beams = args.num_beams
                while True:
                    try:
                        output = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=10,
                            # prefix_allowed_tokens_fn=prefix_allowed_tokens,
                            temperature=1,
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            output_scores=True,
                            return_dict_in_generate=True,
                            early_stopping=True,
                            # logits_processor = logits_processor,
                            pad_token_id=tokenizer.pad_token_id,  # 显式设置 pad_token_id
                            eos_token_id=tokenizer.eos_token_id   # 如果需要也可以显式设置 eos_token_id
                        )
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        print("Out of memory!")
                        num_beams = num_beams - 1
                        print("Beam:", num_beams)
                    except Exception:
                        raise RuntimeError

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                # output_ids = [output_id[-5:-1] for output_id in output_ids.cpu().tolist()]

                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                # # print(output)


                all_device_topk_res = get_topk_results(output, scores, targets, num_beams,
                                            all_items=all_items if args.filter_items else None)
                total += len(all_device_topk_res)
                # bs_gather_list = [None for _ in range(world_size)]
                # dist.all_gather_object(obj=bs, object_list=bs_gather_list)
                # total += sum(bs_gather_list)
                # res_gather_list = [None for _ in range(world_size)]
                # dist.all_gather_object(obj=topk_res, object_list=res_gather_list)
                #
                # all_device_topk_res = []
                # for ga_res in res_gather_list:
                #     all_device_topk_res += ga_res
                batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if (step + 1) % 20 == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)

        for m in metrics_results:
            metrics_results[m] = metrics_results[m] / total

        all_prompt_results.append(metrics_results)
        print("======================================================")
        print("Prompt 0 results: ", metrics_results)
        print("======================================================")
        print("")

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res) / len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")

    save_data = {}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    # if not os.path.exists(args.results_file): os.makedirs(args.results_file)
    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)
    print("Save file: ", args.results_file)



