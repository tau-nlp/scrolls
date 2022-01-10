import os

# from datetime import datetime
from glob import glob
import re

from src.utils.namegenerator import gen as namegenerator

PARENT_OUTPUT_DIR = "outputs"


def get_experiment_name_and_output_dir(
    model_name_or_path, config_name, dataset_name, dataset_config_name, command_split
):
    # time_str = datetime.now().strftime("%Y-%m-%d") + "_" + datetime.now().strftime("%H-%M-%S")
    generated_name = namegenerator(n=2)
    numbered_generated_name = f"{generated_name}-{get_number()}"

    experiment_type_parts = []

    if config_name is not None:
        experiment_type_parts.append(config_name.replace("/", "-"))

    if model_name_or_path is not None and config_name != model_name_or_path:
        if not os.path.exists(model_name_or_path):
            experiment_type_parts.append(model_name_or_path.replace("/", "-"))
        else:
            experiment_type_parts.append(os.path.basename(model_name_or_path).replace("/", "-"))

    command_as_str = " ".join(command_split).replace("=", " ")

    if "--folder_suffix" in command_as_str:
        split_without_eq = command_as_str.split()
        index = split_without_eq.index("--folder_suffix")
        arg_names = split_without_eq[index + 1]
        arg_names_list = arg_names.split("$")
        for arg_name in arg_names_list:
            index = split_without_eq.index(f"--{arg_name}")
            arg_value = split_without_eq[index + 1]
            if arg_name == "global_attention_first_token" and arg_value == "True":
                arg_value = "global"
            experiment_type_parts.append(arg_value)

    experiment_type_parts.append(os.path.splitext(os.path.basename(dataset_name))[0])
    if dataset_config_name is not None:
        experiment_type_parts.append(dataset_config_name)

    name_components = ["_".join(experiment_type_parts), numbered_generated_name]

    output_dir = os.path.join(PARENT_OUTPUT_DIR, "_".join(name_components))

    return numbered_generated_name, output_dir


def get_number():
    directories = [x for x in glob(os.path.join(PARENT_OUTPUT_DIR, "*")) if os.path.isdir(x)]
    number = 0
    number_regex = re.compile(".*-(\d+)")
    for directory in directories:
        regex_result = number_regex.search(os.path.basename(directory))
        if regex_result is not None:
            number = max(number, int(regex_result.group(1)))
    return number + 1
