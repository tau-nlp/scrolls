import sys
import os
import argparse
import json
import re
import shlex
import copy
import collections

from src.utils.output_dir import get_experiment_name_and_output_dir
from src.utils.config import handle_config


def get_command_from_command_parts(command_parts):
    return " ".join(
        map(
            lambda command_part: command_part.strip()[:-1].strip()
            if command_part.strip().endswith("\\")
            else command_part.strip(),
            command_parts,
        )
    )


def get_command_split_from_command(command: str):
    command = command.replace("python -m", "python", 1)
    return shlex.split(command)


def prep_command(command_parts):
    # Setup output directory
    command_str = get_command_from_command_parts(command_parts)
    original_args_list = shlex.split(command_str)
    run_py_index = original_args_list.index("src/run.py")
    args_list = original_args_list[run_py_index:]

    if any(args_list[1].endswith(ext) for ext in [".json"]):
        sys.argv = args_list
        file_args = handle_config()
        args = copy.deepcopy(file_args)
        args_list = sys.argv
    else:
        args = {}

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parsed_args, unknown = parser.parse_known_args(args_list)
    for key, value in vars(parsed_args).items():
        if value is not None:
            args[key] = value

    model_name = args.get("config_name", None)
    if args.get("model_name_or_path", None) is not None:
        if not os.path.isfile(args["model_name_or_path"]):
            model_name = args["model_name_or_path"]
        else:
            with open(os.path.join(os.path.dirname(args["model_name_or_path"]), "config.json"), mode="r") as f:
                model_name = json.load(f)["_name_or_path"]
    assert model_name is not None and len(model_name) > 0

    command_split = (
        original_args_list[: run_py_index + 1]
        + [
            element
            for arg in (
                [f"--{k}", " ".join(v) if isinstance(v, list) else str(v)]
                for k, v in collections.OrderedDict(sorted(args.items())).items()
                if v is not None and (not hasattr(v, "__len__") or len(v) > 0)
            )
            for element in arg
        ]
        + args_list[1:]
    )

    # Remove duplicates
    keys_to_position = {}
    positions = []
    for i, x in enumerate(command_split):
        if x.startswith("--"):
            key = x[2 : (x.index("=") if "=" in x else len(x))]
            keys_to_position[key] = i
            positions.append(i)
    new_command_split = command_split[: positions[0]]
    for key, position in keys_to_position.items():
        next_position = positions[positions.index(position) + 1] if position != positions[-1] else len(command_split)
        new_command_split.extend(command_split[position:next_position])
    command_split = new_command_split

    if "output_dir" not in args:
        experiment_name, output_dir = get_experiment_name_and_output_dir(
            args.get("model_name_or_path", None),
            args.get("config_name", None),
            args["dataset_name"],
            args.get("dataset_config_name", None),
            command_split,
        )
        command_split += ["--output_dir", os.path.abspath(output_dir)]
    else:
        output_dir = args["output_dir"]

    if "run_name" not in args:
        command_split += ["--run_name", os.path.basename(output_dir)]

    print(os.path.abspath(output_dir))

    return {
        "command_parts": command_split,
        "output_dir": output_dir,
    }

