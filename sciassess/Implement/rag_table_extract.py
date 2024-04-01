from typing import Optional

import numpy as np
from evals.api import CompletionFn
from evals.record import RecorderBase

from sciassess.Implement.match_with_func import MatchWithFunc
from sciassess.Implement.utils.metrics import tableMatching
from sciassess.Implement.utils.postprocess import extract_table
from sciassess.Implement.utils.storage import update_dataset_files


class TableExtract(MatchWithFunc):
    def __init__(
            self,
            completion_fns: list[CompletionFn],
            samples_jsonl: str,
            instructions: Optional[str] = "",
            *args,
            **kwargs,
    ):
        super().__init__(completion_fns, samples_jsonl, *args, instructions=instructions, **kwargs)
        assert len(completion_fns) < 3, "TableExtract only supports 3 completion fns"
        self.samples_jsonl = samples_jsonl
        self.instructions = instructions

        self.func_postprocess_answer = extract_table
        self.func_comparison = tableMatching

    # Since we have pointed out the function to be used for postprocessing and comparison,
    # we don't need to override eval_sample() method.

    def run(self, recorder: RecorderBase):
        raw_samples = self.get_samples()
        samples = update_dataset_files(raw_samples)
        self.recorder = recorder
        for sample in samples:
            sample["compare_fields"] = [field if type(field) == str else tuple(field) for field in sample["compare_fields"]]
            sample["input"] = self.instructions
            if "index" not in sample:
                sample["index"] = ("Compound", "")

        self.eval_all_samples(recorder, samples)

        all_sample_metrics = recorder.get_metrics()

        record_metrics = {key: np.mean([sample_metrics[key] for sample_metrics in all_sample_metrics]) for key in
                   ["recall_field", "recall_index", "recall_value", "recall_value_strict", "accuracy_value", "accuracy_value_strict"]}
        if "SMILES" in raw_samples[0]["compare_fields"]:
            record_metrics["recall_SMILES"] = np.mean([sample_metrics["recall_SMILES"]
                                                       for sample_metrics in all_sample_metrics
                                                       if "recall_SMILES" in sample_metrics])
        return record_metrics
