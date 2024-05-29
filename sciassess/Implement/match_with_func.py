from typing import Any, Optional

from sciassess.Implement.utils.storage import update_dataset_files, prepare_few_shot
from sciassess.Implement.completion_fns.utils import ErrorCompletionResult
import evals
import evals.metrics
from evals.api import CompletionFn
from evals.prompt.base import is_chat_prompt
from evals.utils.misc import make_object
from typing import Union
import os

def check_match(
    sampled: str,
    expected: Union[str, list[str], tuple[str]],
) -> bool:
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]

    picked = None
    for option in expected:
        if not sampled.startswith(option):
            continue
        picked = option
        break
    match = picked is not None
    return match

class MatchWithFunc(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 5000,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        instructions: Optional[str] = "",

        func_postprocess_answer: str = None,
        func_comparison: str = None,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        # assert len(completion_fns) == 1, "Match only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl
        self.num_few_shot = num_few_shot
        if self.num_few_shot > 0:
            assert few_shot_jsonl is not None, "few shot requires few shot sample dataset"
            self.few_shot_jsonl = few_shot_jsonl
            self.few_shot = evals.get_jsonl(self._prefix_registry_path(self.few_shot_jsonl))
        self.instructions = instructions

        self.func_postprocess_answer = make_object(func_postprocess_answer) if func_postprocess_answer else None
        self.func_comparison = make_object(func_comparison) if func_comparison else None

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"
        assert isinstance(sample["ideal"], str) or isinstance(
            sample["ideal"], list
        ), "sample['ideal'] must be a string or list of strings"

        prompt = sample["input"]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            **{k: v for k, v in sample.items() if k.startswith("file")}
        )
        sampled = result.get_completions()[0].strip()
        is_fail = isinstance(result, ErrorCompletionResult)

        if hasattr(result, "extras") and "extracted_answer" in result.extras:
            sampled = result.extras["extracted_answer"].rstrip(".")

        ideal = sample["ideal"][0] if isinstance(sample["ideal"], list) else sample["ideal"]
        if is_fail:
            metric = self.func_comparison(ideal, "", **sample) if self.func_comparison else False
            self.recorder.record_error(
                traceback=sampled,
                prompt=sample["input"],
                file_name=sample.get('file_name', None),
                zero_metric=metric
            )
            return
        if self.func_postprocess_answer:
            ideal = self.func_postprocess_answer(ideal, **sample)
            sampled = self.func_postprocess_answer(sampled, **sample)

            if sampled is None or ideal is None:
                metric = self.func_comparison(ideal, "", **sample) if self.func_comparison else False
                self.recorder.record_sample(
                    expected=sample["ideal"],
                    result=sampled,
                    prompt=sample["input"],
                    metric=metric,
                    file_name=sample.get('file_name', None)
                )
                return

        if self.func_comparison:
            metrics = self.func_comparison(ideal, sampled, **sample)
            self.recorder.record_sample(
                expected=sample["ideal"],
                result=sampled,
                prompt=sample["input"],
                metric=metrics,
                file_name=sample.get('file_name', None)
            )
        else:
            match = check_match(sampled, ideal)
            self.recorder.record_sample(
                expected=sample["ideal"],
                result=sampled,
                prompt=sample["input"],
                metric=match,
                file_name=sample.get('file_name', None)
            )

    def run(self, recorder):
        raw_samples = self.get_samples()
        n_shot_path = os.path.join(os.path.dirname(self.samples_jsonl), 'n_shot.jsonl')
        # few_shot
        try:
            self.samples_jsonl = n_shot_path
            n_shot_samples = self.get_samples()
            raw_samples = prepare_few_shot(raw_samples, n_shot_samples)
        except:
            pass
        samples = update_dataset_files(raw_samples)
        for sample in samples:
            if "input" not in sample:
                sample["input"] = self.instructions
        self.recorder = recorder
        self.eval_all_samples(recorder, samples)

        events = recorder.get_events("match")
        if len(events) > 0:
            record_metrics = {
                "accuracy": evals.metrics.get_accuracy(events),
                "bootstrap_std": evals.metrics.get_bootstrap_accuracy_std(events),
            }
        else:
            record_metrics = {}

        # for custom metrics
        all_sample_metrics = recorder.get_metrics()
        if all_sample_metrics and len(all_sample_metrics) > 0:
            metrics = {}
            for metric in all_sample_metrics[0].keys():
                metrics[metric] = 0
                for sample_metrics in all_sample_metrics:
                    if metric not in sample_metrics:
                        continue
                    metrics[metric] += sample_metrics[metric]
                metrics[metric] = metrics[metric] / len(all_sample_metrics)
            record_metrics.update(metrics)

        return record_metrics
