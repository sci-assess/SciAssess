from evals.record import RecorderBase
from pathlib import Path
import wandb
import json
import os
from io import StringIO
from pandas import DataFrame
from sciassess.Implement.utils import PROJECT_PATH
from typing import List, Dict, Union
import logging
logger = logging.getLogger(__name__)
import threading
from copy import deepcopy
global_lock = threading.Lock()

def df2csv(df):
    # for tableMatching
    csv = StringIO()
    df.to_csv(csv, index=False)
    return csv.getvalue()

class SciRecorder(RecorderBase):
    def __init__(self, run_spec):
        super().__init__(run_spec)
        self.count = 0
        self.run_id = run_spec.run_id
        self.completion_fns = str(run_spec.completion_fns[0] if isinstance(run_spec.completion_fns, list) else run_spec.completion_fns)
        self.eval = run_spec.eval_name.split('.')[0]
        self.project_name = run_spec.run_config['project_name']
        if len(self.project_name) == 0:
            self.project_name = f"{self.completion_fns}"
        self.id = f"{self.eval}"
        self.log_path = \
            (f"{PROJECT_PATH}/SciAssess_library/logs/eval_records/{self.completion_fns}/"
             f"{run_spec.run_id}-{self.completion_fns}-{self.eval}.jsonl")
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self.result_path = \
            (f"{PROJECT_PATH}/SciAssess_library/logs/eval_results/{self.completion_fns}/"
             f"{run_spec.run_id}-{self.completion_fns}-{self.eval}-result.json")
        Path(self.result_path).parent.mkdir(parents=True, exist_ok=True)
        self.count_first_fail = 0
        # try:
        #     wandb.ensure_configured()
        #     if not wandb.api.api_key:
        #         raise ValueError("Wandb login not detected. Please run 'wandb login'.")
        #     wandb.init(project=self.project_name, name=self.eval, reinit=True)
        #     wandb.config.completion_fns = self.completion_fns
        #     wandb.config.run_id = self.run_id
        #     wandb.config.eval = self.eval
        #     self.results_table = wandb.Table(columns=["expected", "result", "prompt", "metric", "file_name"])
        #     self.fail_table = wandb.Table(columns=["traceback", "prompt", "file_name"])
        # except (ValueError, wandb.errors.UsageError) as e:
        #     logger.info("wandb not be used. Please login wandb first.")

    def record_error(self,
                     traceback: str,
                     prompt: Union[str, List[Dict]],
                     file_name,
                     zero_metric: Union[bool, Dict] = None):
        sample_record = {
            "status": "error",
            "traceback": traceback,
            "prompt": prompt,
            "file_name": file_name
        }
        with global_lock:
            with open(self.log_path, 'a+') as f:
                f.write(json.dumps(sample_record) + '\n')
            # wandb
            # self.fail_table.add_data(traceback, prompt, file_name)
        if isinstance(zero_metric, bool):
            super().record_match(
                correct=False, expected='None', picked='None', prompt=prompt)
        else:
            super().record_metrics(**zero_metric)
        self.count += 1

    def record_sample(self,
                      expected: Union[str, List[str]],
                      result: Union[str, List[str]],
                      prompt: Union[str, List[Dict]],
                      metric: Union[bool, Dict],
                      **kwargs):
        # For tableMatching
        if isinstance(expected, DataFrame):
            expected = df2csv(expected)
        if isinstance(result, DataFrame):
            result = df2csv(result)

        sample_record = {
            "status": "success",
            "expected": expected,
            "result": result,
            "prompt": prompt,
            "metric": metric,
        }
        sample_record.update(kwargs)
        with global_lock:
            with open(self.log_path, 'a+') as f:
                f.write(json.dumps(sample_record) + '\n')

            # wandb
            # self.results_table.add_data(str(expected), str(result), prompt, metric, kwargs.get('file_name', 'None'))
        if isinstance(metric, bool):
            super().record_match(
                correct=metric, expected=str(expected), picked=str(result), prompt=prompt)
        else:
            super().record_metrics(**metric)
        self.count += 1


    def record_final_report(self, final_report):
        final_report['num_samples'] = self.count
        # wandb.log({'samples': self.results_table})
        # wandb.log({'failed_samples': self.fail_table})
        # wandb.log({'final_result': final_report})
        with open(self.result_path, 'w') as f:
            json.dump(final_report, f)
        super().record_final_report(final_report)
        # wandb.finish()



