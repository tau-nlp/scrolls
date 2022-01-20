# SCROLLS - Prepare Submission File

A script for preparing a SCROLLS submission file.
***

### Requirements
* [Setup](https://github.com/tau-nlp/scrolls/blob/main/evaluator/README.md#setup) environment.
* [Prediction Format](https://github.com/tau-nlp/scrolls/blob/main/evaluator/README.md#prediction-format) expected by the script below.

***

Please set:
* `{dataset_name}_PREDS_FILE` to be the path to a file in the format described in [Predictions Format](#https://github.com/tau-nlp/scrolls/blob/main/evaluator/README.md#prediction-format) containing your predictions for `{dataset_name}`.
  
* `OUTPUT_DIR` to be the path you want the submission file will be saved to.

Run:

```python
python prepare_submission.py \
--gov_report_file GOV_REPORT_PREDS_FILE \
--summ_screen_file SUMM_SCREEN_FD_PREDS_FILE \
--qmsum_file QMSUM_PREDS_FILE \
--narrative_qa_file NARRATIVE_QA_PREDS_FILE \
--qasper_file QASPER_PREDS_FILE \
--quality_file QUALITY_PREDS_FILE \
--contract_nli_file CONTRACT_NLI_PREDS_FILE \
--output_dir OUTPUT_DIR
```

To verify your output file is valid, please see [Verify Submission File](https://github.com/tau-nlp/scrolls/blob/main/evaluator/VERIFY_SUBMISSION_FILE.md).

Upload the file to the [SCROLLS website](https://www.scrolls-benchmark.com) (you may need to login first).
