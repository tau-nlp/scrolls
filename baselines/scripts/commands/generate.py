import os
from scripts.utils import prep_command
from scripts.commands.consts import *


def get_command(id_):
    os.environ["DEBUG"] = os.environ.get("DEBUG", "false")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # fmt: off

    commands_dict = {}
    split_id = id_.split("$$$")
    checkpoint_path = split_id[1]
    id_ = split_id[0]
    num_gpus = 1

    fb_256_bart_args = [
        f"--max_source_length 256",
        f"--max_target_length {FB_BART_MAX_LEN}",
        f"--fp16 {FB_BART_FP16}",
        f"--per_device_eval_batch_size {FB_BART_per_device_eval_batch_size}",
    ]
    fb_512_bart_args = [
        f"--max_source_length 512",
        f"--max_target_length {FB_BART_MAX_LEN}",
        f"--fp16 {FB_BART_FP16}",
        f"--per_device_eval_batch_size {FB_BART_per_device_eval_batch_size}",
    ]
    fb_1024_bart_args = [
        f"--max_source_length {FB_BART_MAX_LEN}",
        f"--max_target_length {FB_BART_MAX_LEN}",
        f"--fp16 {FB_BART_FP16}",
        f"--per_device_eval_batch_size {FB_BART_per_device_eval_batch_size}",
    ]

    allenai_led_args = [
        f"--attention_window {ALLEN_AI_ATTENTION_WINDOW}",
        f"--max_target_length {ALLEN_AI_MAX_TARGET_LEN}",
        f"--fp16 {ALLEN_AI_FP16}",
        f"--per_device_eval_batch_size {ALLEN_AI_per_device_eval_batch_size}",
    ]
    distributed_str = f"-m torch.distributed.run --nproc_per_node={num_gpus}" if num_gpus > 1 else ""
    for dataset in ["qasper", "narrative_qa", "gov_report", "summ_screen_fd", "qmsum", "contract_nli", "quality", "quality_difficult"]:
        base_args = [f"python {distributed_str} src/run.py",
                     f"configs/datasets/{dataset}.json",
                     f"--model_name_or_path {checkpoint_path}",
                     "--m configs/no_metrics.json",
                     "--logging_steps 10",
                     "--preprocessing_num_workers 1",
                     "--predict_with_generate True",
                     "--drop_duplicates_in_eval True",
                     "--num_beams 1",
                     ]

        if dataset == "narrative_qa":
            base_args.append("--trim_very_long_strings")

        commands_dict[f"{dataset}_256-bart_validation"] = base_args + fb_256_bart_args + ["--do_eval True"]
        commands_dict[f"{dataset}_256-bart_test"] = base_args + fb_256_bart_args + ["--do_predict True"]

        commands_dict[f"{dataset}_512-bart_validation"] = base_args + fb_512_bart_args + ["--do_eval True"]
        commands_dict[f"{dataset}_512-bart_test"] = base_args + fb_512_bart_args + ["--do_predict True"]

        commands_dict[f"{dataset}_1024-bart_validation"] = base_args + fb_1024_bart_args + ["--do_eval True"]
        commands_dict[f"{dataset}_1024-bart_test"] = base_args + fb_1024_bart_args + ["--do_predict True"]

        commands_dict[f"{dataset}_led-1024_validation"] = base_args + allenai_led_args + ["--do_eval True",
                                                                                                   "--global_attention_first_token True",
                                                                                                   "--max_source_length 1024"
                                                                                                   ]
        commands_dict[f"{dataset}_led-1024_test"] = base_args + allenai_led_args + ["--do_predict True",
                                                                                                  "--global_attention_first_token True",
                                                                                                  "--max_source_length 1024"]

        commands_dict[f"{dataset}_led-4096_validation"] = base_args + allenai_led_args + ["--do_eval True",
                                                                                                   "--global_attention_first_token True",
                                                                                                   "--max_source_length 4096"
                                                                                                   ]
        commands_dict[f"{dataset}_led-4096_test"] = base_args + allenai_led_args + ["--do_predict True",
                                                                                                  "--global_attention_first_token True",
                                                                                                  "--max_source_length 4096"]

        commands_dict[f"{dataset}_led-16384_validation"] = base_args + allenai_led_args + ["--do_eval True",
                                                                                                   "--global_attention_first_token True",
                                                                                                   "--max_source_length 16384"
                                                                                                   ]
        commands_dict[f"{dataset}_led-16384_test"] = base_args + allenai_led_args + ["--do_predict True",
                                                                                                  "--global_attention_first_token True",
                                                                                                  "--max_source_length 16384"]

    command_parts = commands_dict[id_]
    # fmt: on

    return prep_command(command_parts)
