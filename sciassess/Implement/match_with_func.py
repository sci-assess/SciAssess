from typing import Any, Optional

from sciassess.Implement.utils.storage import update_dataset_files

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.prompt.base import is_chat_prompt
from evals.utils.misc import make_object


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
        assert len(completion_fns) == 1, "Match only supports one completion fn"
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
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            **{k: v for k, v in sample.items() if k.startswith("file")}
        )
        sampled = result.get_completions()[0].strip()

        extras = {"file_name": sample["file_name"]} if "file_name" in sample else {}
        if hasattr(result, "extras"):
            if "extracted_answer" in result.extras:
                sampled = result.extras["extracted_answer"].rstrip(".")
            extras = result.extras
        else:
            extras["answer"] = sampled

        ideal = sample["ideal"][0] if isinstance(sample["ideal"], list) else sample["ideal"]
        if self.func_postprocess_answer:
            extras["answer"] = sampled
            sampled = extras["extracted_answer"] = self.func_postprocess_answer(sampled, **sample)
            ideal = self.func_postprocess_answer(ideal, **sample)

            if sampled is None or ideal is None:
                evals.record.record_match(correct=False,
                                          expected=str(ideal),
                                          picked=str(sampled), sampled=extras["answer"],
                                          prompt=prompt,
                                          **extras)
                return

        if self.func_comparison:
            metrics = self.func_comparison(ideal, sampled, **sample)
            if type(metrics) == bool:
                evals.record.record_match(correct=metrics,
                                          expected=str(ideal),
                                          picked=str(sampled), sampled=extras["answer"],
                                          prompt=prompt,
                                          **extras)
            else:
                evals.record.record_metrics(**metrics)

        else:
            return evals.record_and_check_match(
                prompt=prompt,
                sampled=sampled,
                expected=sample["ideal"],
                # **extras
            )

    def run(self, recorder):
        raw_samples = self.get_samples()
        samples = update_dataset_files(raw_samples)
        for sample in samples:
            if "input" not in sample:
                sample["input"] = self.instructions
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
                    metrics[metric] += sample_metrics[metric]
                metrics[metric] = metrics[metric] / len(all_sample_metrics)
            record_metrics.update(metrics)

        return record_metrics
