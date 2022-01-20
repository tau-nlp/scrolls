import argparse
from time import time
import os
import sys
import json
from collections import namedtuple
import pandas as pd
import logging

from dataset_evaluator import main as evaluate_dataset, get_metrics_filename, DATASETS

log = logging.getLogger(__name__)

EXPECTED_DF_COLS = {"Task", "ID", "Prediction"}
EXPECTED_TASKS = [
    "gov_report",
    "summ_screen_fd",
    "qmsum",
    "narrative_qa",
    "qasper",
    "quality",
    "contract_nli",
]  # in the df
assert set(EXPECTED_TASKS).issubset(DATASETS)
BenchmarkEvaluatorArgs = namedtuple(
    "BenchmarkEvaluatorArgs",
    "all_predictions test_data_dir verify_only split cache_dir metrics_output_dir internal_call",
)
DatasetEvaluatorArgs = namedtuple(
    "DatasetEvaluatorArgs",
    "predictions dataset_name test_data_file verify_only split cache_dir metrics_output_dir internal_call",
)


def main(args):
    all_predictions = args.all_predictions
    if isinstance(all_predictions, str):
        all_predictions = load_predictions_df(all_predictions)

    scrolls_metrics = {}
    for task in EXPECTED_TASKS:
        t0 = time()

        dataset_names = [task] if not (task == "quality" and args.split == "test") else [task, task + "_hard"]

        for dataset_name in dataset_names:
            log.info(f"Evaluating the results for task {task} with datsetname {dataset_name}...")
            task_json = (
                all_predictions[all_predictions.Task == task][["ID", "Prediction"]]
                .set_index("ID")["Prediction"]
                .to_dict()
            )
            evaluator_obj = DatasetEvaluatorArgs(
                predictions=task_json,
                dataset_name=dataset_name,
                test_data_file=os.path.join(args.test_data_dir, task, "test_with_output.jsonl")
                if args.test_data_dir != None
                else None,
                verify_only=args.verify_only,
                split=args.split,
                cache_dir=args.cache_dir,
                metrics_output_dir=args.metrics_output_dir,
                internal_call=True,
            )
            try:
                evaluate_dataset(evaluator_obj, raise_on_errors=True)
            except Exception as e:
                if args.internal_call:
                    return {"e": e, "task": task, "error": "format"}
                else:
                    log.exception(f"Failed to process task: {task} for submission")
                    raise ValueError(f"Format of submission for task {task} is invalid: {e}")

            if args.verify_only:
                continue

            metric_file = get_metrics_filename(args.metrics_output_dir, evaluator_obj.dataset_name)
            if not os.path.exists(metric_file):
                if args.internal_call:
                    return {"task": task, "error": "runtime"}
                else:
                    log.error(f"Internal error in task {task} for submission")
                    raise RuntimeError(f"Failed to compute scores for task {task}.")

            with open(metric_file, "r") as f:
                metrics = json.load(f)

            if not dataset_name.endswith("_hard"):  # dealing with quality
                scrolls_metrics[task] = {
                    "scrolls_score": metrics["scrolls_score"],
                    "display": metrics["display"],
                    "display_keys": metrics["display_keys"],
                }
            else:
                scrolls_metrics[task]["display"] += metrics["display"]
                scrolls_metrics[task]["display_keys"] += [
                    f"hard_{display_key}" for display_key in metrics["display_keys"]
                ]

            log.info(
                f"Finished evaluating task {task} on dataset name {dataset_name} in {time()-t0:.1f} seconds with scores: {metrics}"
            )
            assert len(metrics) > 0

    if args.verify_only:
        if not args.internal_call:
            print("The verification was succesful.")
        sys.exit()

    log.info(f"Computing the aggregated score")
    scrolls_metrics["scrolls_score"] = sum([scrolls_metrics[task]["scrolls_score"] for task in EXPECTED_TASKS]) / len(
        EXPECTED_TASKS
    )

    with open(os.path.join(args.metrics_output_dir, "scrolls.json"), "w") as f:
        json.dump(scrolls_metrics, f, indent=4)

    if args.internal_call:
        return scrolls_metrics
    else:
        print(json.dumps(scrolls_metrics, indent=4))


def load_predictions_df(file_path):
    try:
        df = safe_read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read the csv with pandas: {e}")

    cols = set(df.columns)
    if cols != EXPECTED_DF_COLS:
        raise ValueError(f"csv file has invalid format. Expected columns {EXPECTED_DF_COLS} and got {cols} instead")

    tasks = set(df.Task.unique())
    if tasks != set(EXPECTED_TASKS):
        raise ValueError(
            f"csv file does not contain predictions for the expected tasks. "
            f"Expected tasks {sorted(EXPECTED_TASKS)} and got {sorted(tasks)} instead"
        )

    return df


def safe_read_csv(file_path):
    # https://stackoverflow.com/a/33952294
    return pd.read_csv(file_path, dtype=object, keep_default_na=False, na_values=["!@#$%^&*()"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the predictions for the full SCROLLS benchmark")
    parser.add_argument(
        "--all_predictions",
        type=str,
        help="Path to the file with all of the predictions or the actual predictions",
        required=True,
    )
    parser.add_argument("--metrics_output_dir", type=str, help="Directory of the output metrics file", required=True)
    parser.add_argument("--split", type=str, help="The split to evaluate on", default="test")
    parser.add_argument("--internal_call", type=str, help="For internal use", default=False)
    parser.add_argument(
        "--test_data_dir", type=str, help="Defining the path to the test dir containing the answer files", default=None
    )
    parser.add_argument(
        "--cache_dir", type=str, help="Cache dir for the dataset download", default=None, required=False
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Don't evaluate, just verify that the format and ids are correct.",
        default=False,
    )
    args = parser.parse_args()

    main(args)
