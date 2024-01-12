#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys

sys.path.insert(0, os.getcwd())

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
from copy import deepcopy

import datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import HfArgumentParser
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers.decoding import decode

from src.utils.config import handle_config, save_args, handle_args_to_ignore
from datasets import load_dataset
from src.metrics import load_metric
from src.utils.reproducibility import save_git_info
from src.utils.duplicates import drop_duplicates_in_input

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    drop_duplicates_in_eval: bool = field(
        default=True,
    )
    use_swed: bool = field(
        default=False,
    )
    swed_context_size: Optional[int] = field(
        default=None,
    )
    attention_window: Optional[int] = field(
        default=None,
        metadata={"help": "The attention window size for models such as LED, will be used in the model config"},
    )
    relative_attention_num_buckets: Optional[int] = field(
        default=None,
        metadata={"help": "The relative_attention_num_buckets for T5, will be used in the model config"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use gradient checkpointing in models such ash LED"},
    )
    remove_global_attention: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to remove global attention weight matrices in LED"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library) or name of the file in src/data."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    metric_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The name of the metric to use (from src/metrics)."},
    )
    input_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    output_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Defining the data_dir of the dataset configuration."},
    )
    download_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Defining the download_mode when loading the dataset. Options are `reuse_dataset_if_exists` (default), `reuse_cache_if_exists` and `force_redownload`."
        },
    )
    evaluate_on_training_data: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate on training data or not, to make sure the model can overfit."},
    )
    folder_suffix: str = field(
        default="",
        metadata={"help": "args to be suffixes for the output folder of the run"},
    )
    preprocess_only: bool = field(
        default=False,
        metadata={"help": "Preprocess only: Don't start training, just do the things before"},
    )
    assign_zero_to_too_long_val_examples: bool = field(
        default=False,
        metadata={
            "help": "If true, all sequences longer then max_source_length will be assign a score of 0 in the metric evaluation"
        },
    )
    global_attention_first_token: bool = field(
        default=False,
        metadata={"help": "If true, will assign global attention to first token of all inputs"},
    )
    shared_storage: bool = field(
        default=True,
        metadata={"help": "Whether nodes share the same storage"},
    )
    trim_very_long_strings: bool = field(
        default=False,
        metadata={"help": "Whether to trim very long strings before tokenizing them"},
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    handle_args_to_ignore(sys.argv)  # Just for sweeps

    os.environ["WANDB_WATCH"] = "false"

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if any(sys.argv[1].endswith(ext) for ext in [".json"]):
        config = handle_config()
        model_args, data_args, training_args = parser.parse_dictionary_and_args(config)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.process_index == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_git_info(training_args.output_dir)
        save_args({**asdict(model_args), **asdict(data_args), **training_args.to_dict()})

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Used to find missing dependencies early on
    load_metric(data_args.metric_names, **locals())

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        if not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)

            if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `input_column` and `output_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    data_files = None
    if data_args.train_file is not None or data_args.validation_file is not None or data_args.test_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file

    # Downloading and loading a dataset from the hub/local script.
    seq2seq_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        ignore_verifications=True,
        cache_dir=model_args.cache_dir,
        data_dir=data_args.data_dir,
        data_files=data_files,
        download_mode=data_args.download_mode,
    )

    if data_args.evaluate_on_training_data:
        seq2seq_dataset["validation"] = seq2seq_dataset["train"]

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_name = None
    if model_args.config_name:
        config_name = model_args.config_name
    else:
        if os.path.isfile(model_args.model_name_or_path):
            config_name = os.path.dirname(model_args.model_name_or_path)
        else:
            config_name = model_args.model_name_or_path

    config_overrides = {}
    if model_args.attention_window is not None:
        config_overrides["attention_window"] = model_args.attention_window
    if model_args.gradient_checkpointing is not None:
        config_overrides["gradient_checkpointing"] = model_args.gradient_checkpointing
    if model_args.relative_attention_num_buckets is not None:
        config_overrides["relative_attention_num_buckets"] = model_args.relative_attention_num_buckets
    if model_args.remove_global_attention is not None:
        config_overrides["remove_global_attention"] = model_args.remove_global_attention

    config = AutoConfig.from_pretrained(
        config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        **config_overrides,
    )
    if model_args.model_name_or_path is None:
        # Padding for divisibility by 8
        if config.vocab_size % 8 != 0 and training_args.fp16_padding:
            config.vocab_size += 8 - (config.vocab_size % 8)

    tokenizer_name = None
    if model_args.tokenizer_name:
        tokenizer_name = model_args.tokenizer_name
    else:
        if os.path.isfile(model_args.model_name_or_path):
            tokenizer_name = os.path.dirname(model_args.model_name_or_path)
        else:
            tokenizer_name = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.model_name_or_path is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_config(
            config,
        )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = seq2seq_dataset["train"].column_names
    elif training_args.do_eval:
        column_names = seq2seq_dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = seq2seq_dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    if data_args.input_column is None:
        input_column = "input"
    else:
        input_column = data_args.input_column
        if input_column not in column_names:
            raise ValueError(
                f"--input_column' value '{data_args.input_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.output_column is None:
        output_column = "output"
    else:
        output_column = data_args.output_column
        if output_column not in column_names:
            raise ValueError(
                f"--output_column' value '{data_args.output_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function_kwargs_fn():
        return {
            "tokenizer": deepcopy(tokenizer),
            "prefix": prefix,
            "input_column": input_column,
            "output_column": output_column,
            "max_source_length": data_args.max_source_length,
            "max_target_length": max_target_length,
            "padding": padding,
            "ignore_pad_token_for_loss": data_args.ignore_pad_token_for_loss,
            "assign_zero_to_too_long_val_examples": data_args.assign_zero_to_too_long_val_examples,
            "global_attention_first_token": data_args.global_attention_first_token,
            "trim_very_long_strings": data_args.trim_very_long_strings,
        }

    if training_args.do_train:
        if "train" not in seq2seq_dataset:
            raise ValueError("--do_train requires a train dataset")
        logger.info("")
        logger.info("Training examples before tokenization:")
        logger.info(f"input #0: {seq2seq_dataset['train'][0]['input']}")
        logger.info(f"output #0: {seq2seq_dataset['train'][0]['output']}")
        logger.info(f"input #1: {seq2seq_dataset['train'][1]['input']}")
        logger.info(f"output #1: {seq2seq_dataset['train'][1]['output']}")
        logger.info("")
        untokenized_train_dataset = seq2seq_dataset["train"]
        if data_args.max_train_samples is not None:
            untokenized_train_dataset = untokenized_train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(
            local=not data_args.shared_storage, desc="train dataset map pre-processing"
        ):
            train_dataset = untokenized_train_dataset.map(
                preprocess_function,
                fn_kwargs=preprocess_function_kwargs_fn(),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=untokenized_train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        preprocess_function_kwargs = preprocess_function_kwargs_fn()
        preprocess_function_kwargs["max_target_length"] = max_target_length
        if "validation" not in seq2seq_dataset:
            raise ValueError("--do_eval requires a validation dataset")
        logger.info("")
        logger.info("Validation examples before tokenization:")
        logger.info(f"input #0: {seq2seq_dataset['validation'][0]['input']}")
        logger.info(f"output #0: {seq2seq_dataset['validation'][0]['output']}")
        logger.info(f"input #1: {seq2seq_dataset['validation'][1]['input']}")
        logger.info(f"output #1: {seq2seq_dataset['validation'][1]['output']}")
        logger.info("")
        untokenized_eval_dataset = seq2seq_dataset["validation"]
        if data_args.max_eval_samples is not None:
            untokenized_eval_dataset = untokenized_eval_dataset.select(range(data_args.max_eval_samples))
        if model_args.drop_duplicates_in_eval is True:
            untokenized_eval_dataset = drop_duplicates_in_input(untokenized_eval_dataset)
        with training_args.main_process_first(
            local=not data_args.shared_storage, desc="validation dataset map pre-processing"
        ):
            eval_dataset = untokenized_eval_dataset.map(
                preprocess_function,
                fn_kwargs=preprocess_function_kwargs,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=untokenized_eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        preprocess_function_kwargs = preprocess_function_kwargs_fn()
        preprocess_function_kwargs["max_target_length"] = max_target_length
        if "test" not in seq2seq_dataset:
            raise ValueError("--do_predict requires a test dataset")
        untokenized_predict_dataset = seq2seq_dataset["test"]
        if data_args.max_predict_samples is not None:
            untokenized_predict_dataset = untokenized_predict_dataset.select(range(data_args.max_predict_samples))
        if model_args.drop_duplicates_in_eval is True:
            untokenized_predict_dataset = drop_duplicates_in_input(untokenized_predict_dataset)

        if output_column in untokenized_predict_dataset.column_names:
            untokenized_predict_dataset = untokenized_predict_dataset.remove_columns(output_column)

        with training_args.main_process_first(
            local=not data_args.shared_storage, desc="prediction dataset map pre-processing"
        ):
            predict_dataset = untokenized_predict_dataset.map(
                preprocess_function,
                fn_kwargs=preprocess_function_kwargs,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=untokenized_predict_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    if data_args.preprocess_only:
        logger.info(f"With --preprocess_only, exiting after preprocess_on the data")
        exit()

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 and training_args.fp16_padding else None,
    )

    # Metric
    compute_metrics = load_metric(data_args.metric_names, **locals())

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        untokenized_eval_dataset=untokenized_eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        output_dir=training_args.output_dir,
        data_args=data_args,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                id_to_prediction = {}
                for i, instance in enumerate(untokenized_predict_dataset):
                    id_to_prediction[instance["id"]] = predict_results.predictions[i]
                predictions = decode(id_to_prediction, tokenizer, data_args)
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
                with open(output_prediction_file, "w") as writer:
                    json.dump(predictions, writer, indent=4)
                logger.info(f"Predictions saved to {output_prediction_file}")

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    return results


def preprocess_function(
    examples,
    tokenizer,
    prefix,
    input_column,
    output_column,
    max_source_length,
    max_target_length,
    padding,
    ignore_pad_token_for_loss,
    assign_zero_to_too_long_val_examples,
    global_attention_first_token,
    trim_very_long_strings,
):
    if not isinstance(examples[input_column][0], str):
        input_ids = []
        attention_mask = []
        for token_ids in examples[input_column]:
            length = len(token_ids)
            if max_source_length is not None:
                length = min(max_source_length, length)
                input_ids.append(token_ids[:length])
            attention_mask.append([1 for _ in range(length)])
        if len(input_ids) == 0:
            input_ids = examples[input_column]
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if max_source_length is not None:
            model_inputs["not_valid_for_eval"] = [
                len(token_ids) > max_source_length and assign_zero_to_too_long_val_examples
                for token_ids in examples[input_column]
            ]
        else:
            model_inputs["not_valid_for_eval"] = [False for token_ids in examples[input_column]]
    else:
        inputs = examples[input_column]
        if prefix != "":
            inputs = [prefix + inp for inp in inputs]
        if trim_very_long_strings:
            inputs = [inp[: max_source_length * 7] for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        if max_source_length is not None and assign_zero_to_too_long_val_examples:
            model_inputs_untrimmed = tokenizer(inputs)
            model_inputs["not_valid_for_eval"] = [
                len(token_ids) > max_source_length for token_ids in model_inputs_untrimmed["input_ids"]
            ]
        else:
            model_inputs["not_valid_for_eval"] = [False] * len(model_inputs["input_ids"])

    targets = examples[output_column] if output_column in examples else None
    if targets is not None:
        if not isinstance(targets[0], str):
            if max_target_length is not None:
                targets = [target[:max_target_length] for target in targets]
            model_inputs["labels"] = targets
        else:
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
    model_inputs["length"] = [len(x) for x in model_inputs["input_ids"]]
    if global_attention_first_token:
        model_inputs["global_attention_mask"] = [
            [1] + [0] * (len(attn_mask) - 1) for attn_mask in model_inputs["attention_mask"]
        ]
    return model_inputs


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
