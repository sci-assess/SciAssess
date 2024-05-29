from sciassess.Implement.utils import PROJECT_PATH
import argparse
import os
import yaml
import json

# get all evals
eval_sets_dir = f"{PROJECT_PATH}/sciassess/Registry/eval_sets"
eval_sets = {}
for file in os.listdir(eval_sets_dir):
    if file.endswith("yaml"):
        with open(f"{eval_sets_dir}/{file}") as f:
            eval_sets.update(yaml.safe_load(f))



# get completion fn from args
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('completion_fn', type=str, help='completion fn name')
args = parser.parse_args()
completion_fn = args.completion_fn

# get eval results file
# TODO: The most recent result for obtaining the specified eval and completion should be modified to search directly by run_id
eval_result_dir = f"{PROJECT_PATH}/SciAssess_library/logs/eval_results"
eval_results_filenames = [os.path.join(eval_result_dir, file) for file in os.listdir(eval_result_dir)]
eval_results_filenames = sorted(eval_results_filenames, key=os.path.getctime)

# get metric for all evals
def get_metric_for_eval(eval):
    for file in os.listdir(f"{PROJECT_PATH}/sciassess/Registry/evals"):
        if file.endswith(f"_{eval}.yaml"):
            with open(f"{PROJECT_PATH}/sciassess/Registry/evals/{file}") as f:
                eval_yaml = yaml.safe_load(f)
            return eval_yaml[eval]['metrics'][0]
    return None


eval_results = {}
count = 0
total_score = 0
eval_count = 0
missing_evals = []
for eval_set in eval_sets:
    eval_results[eval_set] = {}
    for eval in eval_sets[eval_set]['evals']:
        eval_results[eval_set][eval] = "Not found"
        eval_count += 1
        eval_yaml_filename = f"{PROJECT_PATH}/sciassess/Registry/evals/{eval}.yaml"
        metric = get_metric_for_eval(eval)
        if metric is None:
            metric = 'accuracy' # default metric
        for eval_results_filename in eval_results_filenames[::-1]:
            if eval_results_filename.endswith(f"{completion_fn}-{eval}-result.json"):
                with open(eval_results_filename, 'r') as f:
                    result = json.load(f)
                if metric in result:
                    eval_results[eval_set][eval] = "{:.3f}".format(result[metric]) + f" ({metric})"
                    count += 1
                    total_score += result[metric]
                    break
                else:
                    # use the first key
                    if len(result) == 0:
                        metric = "No result"
                    else:
                        metric = list(result.keys())[0]
                    eval_results[eval_set][eval] = "{:.3f}".format(result[metric]) + f" ({metric})"
                    count += 1
                    total_score += result[metric]
                    break
        if eval_results[eval_set][eval] == "Not found":
            missing_evals.append(f"{eval_set}-{eval}")

def highlight(str: str) -> str:
    return f"\033[1;32m{str}\033[0m"

print(highlight('-------------------------------------------------'))
print(highlight(f"Evaluation results for {completion_fn}"))
print()
for category, experiments in eval_results.items():
    print(highlight(f"{category.replace('_', ' ').title()}:"))
    for experiment, result in experiments.items():
        if experiment.startswith('_'):
            experiment = experiment[1:]
        print(highlight(f"\t{experiment.replace('_', ' ').title()}: {result}"))
    print()

print(highlight("\nAverage metric: {:.3f}".format(total_score/count)))
print(highlight('-------------------------------------------------'))

if eval_count != count:
    print("Some eval results are missing. This may be due to the corresponding eval implement failure or unknown file changes. Please check the logs or re-run the eval.")
    print(f"Missing evals: {missing_evals}")
