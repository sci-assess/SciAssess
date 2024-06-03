import os
import random
from pathlib import Path
import requests
from PyPDF2 import PdfReader, PdfWriter
from typing import List

def highlight(str: str) -> str:
    return f"\033[1;32m {str}\033[0m"

def update_dataset_files(raw_samples: List[dict]) -> List[dict]:
    """
    Update the file links (doi) in the dataset to local files.
    Extract a section from a PDF file and save it as a new file if necessary.
    """
    for raw_sample in raw_samples:
        for ftype in ["", "answer"]:
            if f"{ftype}file_name" in raw_sample and Path(raw_sample[f"{ftype}file_name"]).exists():
                # If the file exists locally, do nothing
                continue
            elif f"{ftype}file_link" in raw_sample:
                # If the file exists in OSS but not locally, download it from OSS
                local_file = raw_sample[f"{ftype}file_name"] if f"{ftype}file_name" in raw_sample else \
                    os.path.join('SciAssess_library/tmp/data', os.path.basename(raw_sample[f"{ftype}file_link"]))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                if not os.path.exists(local_file):
                    response = requests.get(raw_sample[f"{ftype}file_link"])
                    assert response.status_code == 200, f"File not found. Please check the link {raw_sample[f'{ftype}file_link']} and contact us if the problem persists"
                    with open(local_file, 'wb') as f:
                        f.write(response.content)
                    print(f"{local_file} downloaded.")
            elif 'doi' in raw_sample:
                # Due to copyright restrictions, we are unable to directly distribute the original PDF of the article.
                # You will need to download the corresponding PDF according to the instructions in README and store it in SciAssess_library/pdfs.
                doi = raw_sample['doi'].replace('/', '_').replace(' (Supporting Information)', '_si')
                local_file = f"SciAssess_library/pdfs/{doi}.pdf"
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                raw_sample['file_name'] = local_file
                assert Path(local_file).exists(), \
                    highlight(f"""Due to copyright restrictions, we are unable to directly distribute the original PDF of the article.
                    You will need to download the corresponding PDF according to the instructions in README and store it in SciAssess_library/pdfs.
                    {local_file} does not exist.""")
                # If the pages are not from 1 to -1, take a screenshot and save it to tmp
                if 'pages' in raw_sample and raw_sample['pages'] != [1, -1]:
                    pages = raw_sample['pages']
                    segmented_local_file = f"SciAssess_library/tmp/segmented_pdfs/{doi}_p{pages[0]}-p{pages[1]}.pdf"
                    os.makedirs(os.path.dirname(segmented_local_file), exist_ok=True)
                    if not Path(segmented_local_file).exists():
                        reader = PdfReader(local_file)
                        writer = PdfWriter()
                        for i in range(pages[0] - 1, pages[1]):
                            writer.add_page(reader.pages[i])

                        with open(segmented_local_file, 'wb') as output_pdf:
                            writer.write(output_pdf)
                    raw_sample['file_name'] = segmented_local_file

        if "answerfile_name" in raw_sample and Path(raw_sample["answerfile_name"]).exists():
            raw_sample["ideal"] = Path(raw_sample["answerfile_name"]).read_text()
        elif "answerfile_name" in raw_sample and not Path(raw_sample["answerfile_name"]).exists():
            print(raw_sample["answerfile_name"])
    return raw_samples

import random
def prepare_few_shot(raw_samples, n_shot_samples):
    n_shot = int(os.environ.get('N_SHOT', 3))
    examples = []
    for sample in n_shot_samples:
        example_input = None
        for item in sample['input']:
            if item['role'] != 'user':
                continue
            example_input = item['content']
            break
        if example_input is not None:
            example_output = sample['ideal']
            examples.append({'input': example_input, 'output': example_output})
    n_shot = min(len(examples), n_shot)
    if n_shot == 0:
        return raw_samples
    for sample in raw_samples:
        n_shot_text = "\nHere is input/output examples:\n"
        for example in random.choices(examples, k=n_shot):
            n_shot_text += "<input>\n"
            n_shot_text += example['input'] + '\n'
            n_shot_text += "<input>\n"
            n_shot_text += "<output>\n"
            n_shot_text += example['output'] + '\n'
            n_shot_text += "<output>\n"
            n_shot_text += '\n'

        if sample['input'][0]['role'] != 'system':
            sample['input'][0]['content'] = '\n\n' + n_shot_text + sample['input'][0]['content']
        else:
            sample['input'][0]['content'] = sample['input'][0]['content'] + n_shot_text

    return raw_samples

