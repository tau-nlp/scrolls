from typing import Any, Dict, List
from dataclasses import dataclass

import argparse
from argparse import Namespace
import json
import os
import sys

from transformers.training_args import trainer_log_levels


def handle_args_to_ignore(args: List[str]):
    indices_to_remove = []
    for i, arg in enumerate(args):
        if "_ignore_" in arg:
            indices_to_remove.append(i)
            if not arg.startswith("-"):
                indices_to_remove.append(i - 1)

    for i in sorted(indices_to_remove, reverse=True):
        del args[i]


def save_args(original_args):
    reversed_trainer_log_levels = {v: k for k, v in trainer_log_levels.items()}
    original_args["log_level"] = reversed_trainer_log_levels[original_args["log_level"]]
    original_args["log_level_replica"] = reversed_trainer_log_levels[original_args["log_level_replica"]]
    for arg in ["_n_gpu", "local_rank"]:
        if arg in original_args:
            del original_args[arg]
    to_delete = []
    for k, v in original_args.items():
        if v is None:
            to_delete.append(k)
        if original_args[k] == "":
            to_delete.append(k)
    for k in to_delete:
        del original_args[k]
    with open(os.path.join(original_args["output_dir"], "args.json"), mode="w") as f:
        json.dump(original_args, f, indent=4)


def handle_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help=".json path")
    parser.add_argument("--m", dest="merge_files", action="append", default=[])
    args, unknown = parser.parse_known_args()

    config_handler = JsonConfigHandler(args, unknown)
    config = config_handler()
    return config


@dataclass
class ConfigHandler:
    args: Namespace
    unknown: List[str]

    def __call__(self) -> Dict[str, Any]:
        config = self._obtain_config()
        config = merge(config, self.args.merge_files)

        sys.argv = [sys.argv[0]] + self.unknown

        self._edit_config(config)

        return config

    def _edit_config(self, config):
        # remove arguments that cannot be passed through the command line
        to_delete = []
        for k, v in config.items():
            if v is None:
                to_delete.append(k)
            if config[k] == "":
                to_delete.append(k)
        for k in to_delete:
            del config[k]


@dataclass
class JsonConfigHandler(ConfigHandler):
    def _obtain_config(self):
        with open(self.args.path, mode="r") as f:
            config = json.load(f)
        return config


def replace_recursive(obj, old, new):
    if isinstance(obj, str):
        return obj.replace(old, new)

    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = replace_recursive(value, old, new)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = replace_recursive(value, old, new)
    return obj


def merge(config: Dict, merge_files):
    for merge_file in merge_files:
        with open(merge_file, mode="r") as f:
            merge_dict = json.load(f)
        config = with_fallback(preferred=merge_dict, fallback=config)

    return config


# Copied from https://github.com/allenai/allennlp/blob/86504e6b57b26bb2bb362e33c0edc3e49c0760fe/allennlp/common/params.py
import copy
import os
from typing import Any, Dict


def unflatten(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a "flattened" dict with compound keys, e.g.
        {"a.b": 0}
    unflatten it:
        {"a": {"b": 0}}
    """
    unflat: Dict[str, Any] = {}

    for compound_key, value in flat_dict.items():
        curr_dict = unflat
        parts = compound_key.split(".")
        for key in parts[:-1]:
            curr_value = curr_dict.get(key)
            if key not in curr_dict:
                curr_dict[key] = {}
                curr_dict = curr_dict[key]
            elif isinstance(curr_value, dict):
                curr_dict = curr_value
            else:
                raise Exception("flattened dictionary is invalid")
        if not isinstance(curr_dict, dict) or parts[-1] in curr_dict:
            raise Exception("flattened dictionary is invalid")
        curr_dict[parts[-1]] = value

    return unflat


def with_fallback(preferred: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dicts, preferring values from `preferred`.
    """

    def merge(preferred_value: Any, fallback_value: Any) -> Any:
        if isinstance(preferred_value, dict) and isinstance(fallback_value, dict):
            return with_fallback(preferred_value, fallback_value)
        elif isinstance(preferred_value, dict) and isinstance(fallback_value, list):
            # treat preferred_value as a sparse list, where each key is an index to be overridden
            merged_list = fallback_value
            for elem_key, preferred_element in preferred_value.items():
                try:
                    index = int(elem_key)
                    merged_list[index] = merge(preferred_element, fallback_value[index])
                except ValueError:
                    raise Exception(
                        "could not merge dicts - the preferred dict contains "
                        f"invalid keys (key {elem_key} is not a valid list index)"
                    )
                except IndexError:
                    raise Exception(
                        "could not merge dicts - the preferred dict contains "
                        f"invalid keys (key {index} is out of bounds)"
                    )
            return merged_list
        # elif isinstance(preferred_value, list) and isinstance(fallback_value, list):
        #     # merge lists instead of replace, which is more consistent with dictionaries merge
        #     return copy.deepcopy(fallback_value) + copy.deepcopy(preferred_value)
        else:
            return copy.deepcopy(preferred_value)

    preferred_keys = set(preferred.keys())
    fallback_keys = set(fallback.keys())
    common_keys = preferred_keys & fallback_keys

    merged: Dict[str, Any] = {}

    for key in preferred_keys - fallback_keys:
        merged[key] = copy.deepcopy(preferred[key])
    for key in fallback_keys - preferred_keys:
        merged[key] = copy.deepcopy(fallback[key])

    for key in common_keys:
        preferred_value = preferred[key]
        fallback_value = fallback[key]

        merged[key] = merge(preferred_value, fallback_value)
    return merged
