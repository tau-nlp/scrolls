from typing import List, Dict, Any
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
import runpy
import importlib
import subprocess
import random
import string

from scripts.utils import get_command_from_command_parts, get_command_split_from_command


def main(command_dict: Dict[str, Any], unknown):
    command_parts: List[str] = command_dict["command_parts"]
    command = get_command_from_command_parts(command_parts)

    command_split = get_command_split_from_command(command)

    # Prepare the command line arguments
    start_index = 1 if command_split[0] == "python" else 0
    if start_index == 1 and not command_split[start_index].endswith(".py"):
        command_split[start_index] = command_split[start_index].replace(".", "/") + ".py"
    sys.argv = command_split[start_index:] + unknown
    program = sys.argv[0]

    # Find the module name
    module_name = program.replace("/", ".").replace(".py", "")

    print(" ".join(sys.argv))
    runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command_path", type=str)
    parser.add_argument("id", type=str)
    parser.add_argument("--checkpoint_path", default="", type=str)
    args, unknown = parser.parse_known_args()

    command_path = args.command_path
    if command_path.endswith(".py"):
        command_path = command_path[:-3]

    command_id = args.id
    if args.checkpoint_path != "":
        command_id = f"{args.id}$$${args.checkpoint_path}"

    command_dict = importlib.import_module(command_path.replace("/", ".")).get_command(command_id)

    main(command_dict, unknown)
