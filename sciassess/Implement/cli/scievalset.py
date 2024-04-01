"""
This file defines the `scievalset` CLI for running eval sets.
"""
import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, cast

from evals.registry import Registry
from evals.cli.oaievalset import Progress, highlight, get_parser, OaiEvalSetArguments
from sciassess.Implement.utils import PROJECT_PATH

Task = list[str]
logger = logging.getLogger(__name__)

def run(
    args: OaiEvalSetArguments,
    unknown_args: list[str],
    registry: Optional[Registry] = None,
    run_command: str = "scieval",
) -> None:
    registry = registry or Registry()
    if args.registry_path:
        registry.add_registry_paths(args.registry_path)

    commands: list[Task] = []
    eval_set = registry.get_eval_set(args.eval_set) if args.eval_set else None
    if eval_set:
        for index, eval in enumerate(registry.get_evals(eval_set.evals)):
            if not eval or not eval.key:
                logger.debug("The eval #%d in eval_set is not valid", index)
                continue

            command: list[str] = [run_command, args.model, eval.key] + unknown_args
            if args.registry_path:
                command.append("--registry_path")
                command = command + args.registry_path
            if command in commands:
                continue
            commands.append(command)
    else:
        logger.warning("No eval set found for %s", args.eval_set)

    num_evals = len(commands)

    progress = Progress(f"{PROJECT_PATH}/SciAssess_library/tmp/scievalset/{args.model}.{args.eval_set}.progress.txt")
    if args.resume and progress.load():
        print(f"Loaded progress from {progress.file}")
        print(f"{len(progress.completed)}/{len(commands)} evals already completed:")
        for item in progress.completed:
            print("  " + " ".join(item))

    commands = [c for c in commands if c not in progress.completed]
    command_strs = [" ".join(cmd) for cmd in commands]
    print("Going to run the following commands:")
    for command_str in command_strs:
        print("  " + command_str)

    num_already_completed = num_evals - len(commands)
    for idx, command in enumerate(commands):
        real_idx = idx + num_already_completed
        print(highlight("Running command: " + " ".join(command) + f" ({real_idx+1}/{num_evals})"))
        subprocess.run(command, stdout=subprocess.PIPE, check=args.exit_on_error)
        progress.add(command)

    print(highlight("All done!"))


def main() -> None:
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    run(cast(OaiEvalSetArguments, args), unknown_args)


if __name__ == "__main__":
    main()
