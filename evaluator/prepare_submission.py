import argparse
import os

import numpy as np
import pandas as pd
import json

SUBMISSION_LINK = "https://scrolls-benchmark.com/submission"
TASKS_MAPPING = {
    "qmsum_file": "qmsum",
    "qasper_file": "qasper",
    "summ_screen_file": "summ_screen_fd",
    "quality_file": "quality",
    "narrative_qa_file": "narrative_qa",
    "contract_nli_file": "contract_nli",
    "gov_report_file": "gov_report",
}
COLUMNS = ["Task", "ID", "Prediction"]


def safe_read_csv(file_path):
    # https://stackoverflow.com/a/33952294
    return pd.read_csv(file_path, dtype=object, keep_default_na=False, na_values=["!@#$%^&*()"])


def main():
    parser = argparse.ArgumentParser(description="Prepare SCROLLS predictions")
    parser.add_argument("--output_dir", type=str, help="Path to output the predictions file", required=True)
    parser.add_argument(
        "--qmsum_file", type=str, help="The path to the qmsum dataset json file containing predictions", required=True
    )
    parser.add_argument(
        "--qasper_file",
        type=str,
        help="The path to the qasper dataset json file containing predictions",
        required=True,
    )
    parser.add_argument(
        "--summ_screen_file",
        type=str,
        help="The path to the summ_screen dataset json file containing predictions",
        required=True,
    )
    parser.add_argument(
        "--quality_file",
        type=str,
        help="The path to the quality dataset json file containing predictions",
        required=True,
    )
    parser.add_argument(
        "--narrative_qa_file",
        type=str,
        help="The path to the narrative_qa dataset json file containing predictions",
        required=True,
    )
    parser.add_argument(
        "--contract_nli_file",
        type=str,
        help="The path to the contact_nli dataset json file containing predictions",
        required=True,
    )
    parser.add_argument(
        "--gov_report_file",
        type=str,
        help="The path to the gov_report dataset json file containing predictions",
        required=True,
    )
    args = parser.parse_args()

    tasks_dfs = pd.DataFrame(columns=COLUMNS, data=[])
    for file_key, task_name in TASKS_MAPPING.items():
        print(f"Adding predictions for {task_name} from {file_key}...")
        with open(getattr(args, file_key)) as f:
            task_data = json.load(f)
        task_df = pd.DataFrame.from_dict(task_data, orient="index", columns=COLUMNS[-1:]).reset_index(drop=False)
        task_df[COLUMNS[0]] = task_name
        task_df[COLUMNS[1]] = task_df["index"]
        tasks_dfs = pd.concat((tasks_dfs, task_df[COLUMNS]))

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, "scrolls_predictions.csv")
    print(f"Saving the complete predictions file to: {outfile}")
    tasks_dfs = tasks_dfs.reset_index(drop=True)
    tasks_dfs.to_csv(outfile, index=False)

    print("validating submission file is exactly the same as expected")
    recovered_tasks_dfs = safe_read_csv(outfile)
    assert len(recovered_tasks_dfs) == len(tasks_dfs)
    assert recovered_tasks_dfs.columns.tolist() == tasks_dfs.columns.tolist()
    assert np.all(recovered_tasks_dfs.values == tasks_dfs.values)

    print(f"Your benchmark predictions file is ready. If it contains predictions for the test sets please head over to {SUBMISSION_LINK} to submit to the SCROLLS leaderboard.")


if __name__ == "__main__":
    main()
