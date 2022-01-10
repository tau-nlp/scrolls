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

from scripts.utils import get_command_from_command_parts, get_command_split_from_command, has_slurm


def main(command_dict: Dict[str, Any], unknown):
    command_parts: List[str] = command_dict["command_parts"]
    command = get_command_from_command_parts(command_parts)

    if has_slurm():
        if len(unknown) > 0:
            command = f"{command} {' '.join(unknown)}"
        output_dir = command_dict["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        sbatch_file = f"/tmp/temp_{''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(5))}.sbatch"
        with open(sbatch_file, mode="w", encoding="utf=8") as f:
            f.write("#!/bin/sh\n")
            f.write(f"#SBATCH --job-name={os.path.basename(output_dir)}\n")
            f.write(f"#SBATCH --output={output_dir}/out.txt\n")
            f.write(f"#SBATCH --error={output_dir}/err.txt\n")
            f.write(f"#SBATCH --time=4320\n")
            f.write(f"#SBATCH --nodes=1\n")
            f.write(f"#SBATCH --ntasks=1\n")
            f.write(f"#SBATCH --mem=500000\n")
            f.write(f"#SBATCH --cpus-per-task=64\n")
            f.write(f"#SBATCH --gres=gpu:8\n")
            f.write(f"#SBATCH --constraint=volta32gb\n")
            f.write(f"srun {command}")
        print(subprocess.check_output(f"sbatch {sbatch_file}", stderr=subprocess.STDOUT, shell=True).decode())
        print(command)
    else:
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
    parser.add_argument("--no_slurm", action="store_true", default=False)
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
