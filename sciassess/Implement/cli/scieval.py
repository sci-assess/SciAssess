"""
This file defines the `scieval` CLI for running evals.
"""
import argparse
import logging
import shlex
import sys
from typing import Any, Mapping, Optional, Union, cast

import evals
import evals.api
import evals.base
import evals.record
from evals.eval import Eval
from evals.record import RecorderBase
from evals.registry import Registry
from sciassess.Implement.utils import PROJECT_PATH
from sciassess.Implement.recorder import SciRecorder

logger = logging.getLogger(__name__)


def _purple(str: str) -> str:
    return f"\033[1;35m{str}\033[0m"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evals through the API")
    parser.add_argument(
        "completion_fn",
        type=str,
        help="One or more CompletionFn URLs, separated by commas (,). A CompletionFn can either be the name of a model available in the OpenAI API or a key in the registry (see evals/registry/completion_fns).",
    )
    parser.add_argument("eval", type=str, help="Name of an eval. See registry.")
    parser.add_argument("--extra_eval_params", type=str, default="")
    parser.add_argument(
        "--completion_args",
        type=str,
        default="",
        help="Specify additional parameters to modify the behavior of the completion_fn during its creation. Parameters should be passed as a comma-separated list of key-value pairs (e.g., 'key1=value1,key2=value2'). This option allows for the dynamic modification of completion_fn settings, including the ability to override default arguments where necessary.",
    )
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20220722)
    parser.add_argument("--user", type=str, default="")
    parser.add_argument(
        "--registry_path",
        type=str,
        default=None,
        action="append",
        help="Path to the registry",
    )
    return parser


class SciEvalArguments(argparse.Namespace):
    completion_fn: str
    eval: str
    extra_eval_params: str
    cache: bool
    seed: int
    user: str
    registry_path: list[str]


def run(args: SciEvalArguments, registry: Optional[Registry] = None) -> str:
    registry = registry or Registry()
    if args.registry_path:
        registry.add_registry_paths(args.registry_path)

    eval_spec = registry.get_eval(args.eval)
    assert (
        eval_spec is not None
    ), f"Eval {args.eval} not found. Available: {list(sorted(registry._evals.keys()))}"

    def parse_extra_eval_params(
        param_str: Optional[str],
    ) -> Mapping[str, Union[str, int, float]]:
        """Parse a string of the form "key1=value1,key2=value2" into a dict."""
        if not param_str:
            return {}

        def to_number(x: str) -> Union[int, float, str]:
            try:
                return int(x)
            except (ValueError, TypeError):
                pass
            try:
                return float(x)
            except (ValueError, TypeError):
                pass
            return x

        str_dict = dict(kv.split("=") for kv in param_str.split(","))
        return {k: to_number(v) for k, v in str_dict.items()}

    extra_eval_params = parse_extra_eval_params(args.extra_eval_params)
    eval_spec.args.update(extra_eval_params)

    # If the user provided an argument to --completion_args, parse it into a dict here, to be passed to the completion_fn creation **kwargs
    completion_args = args.completion_args.split(",")
    additional_completion_args = {k: v for k, v in (kv.split("=") for kv in completion_args if kv)}

    completion_fns = args.completion_fn.split(",")
    completion_fn_instances = [
        registry.make_completion_fn(url, **additional_completion_args) for url in completion_fns
    ]

    run_config = {
        "completion_fns": completion_fns,
        "eval_spec": eval_spec,
        "seed": args.seed,
        "command": " ".join(map(shlex.quote, sys.argv)),
    }

    eval_name = eval_spec.key
    if eval_name is None:
        raise Exception("you must provide a eval name")

    run_spec = evals.base.RunSpec(
        completion_fns=completion_fns,
        eval_name=eval_name,
        base_eval=eval_name.split(".")[0],
        split=eval_name.split(".")[1],
        run_config=run_config,
        created_by=args.user,
    )

    recorder = SciRecorder(run_spec)

    api_extra_options: dict[str, Any] = {}
    if not args.cache:
        api_extra_options["cache_level"] = 0

    run_url = f"{run_spec.run_id}"
    logger.info(_purple(f"Run started: {run_url}"))

    eval_class = registry.get_class(eval_spec)
    eval: Eval = eval_class(
        completion_fns=completion_fn_instances,
        seed=args.seed,
        name=eval_name,
        eval_registry_path=eval_spec.registry_path,
        registry=registry,
        **extra_eval_params,
    )
    result = eval.run(recorder)
    recorder.record_final_report(result)

    logger.info("Final report:")
    for key, value in result.items():
        logger.info(f"{key}: {value}")
    return run_spec.run_id

def main() -> None:
    parser = get_parser()
    args = cast(SciEvalArguments, parser.parse_args(sys.argv[1:]))
    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        filename=None # TODO: add log file and stdout handler
    )
    logging.getLogger("openai").setLevel(logging.WARN)

    run(args)


if __name__ == "__main__":
    main()
