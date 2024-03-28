import os
from pathlib import Path
import requests
from PyPDF2 import PdfReader, PdfWriter

def highlight(str: str) -> str:
    return f"\033[1;32m {str}\033[0m"

def update_dataset_files(raw_samples: list[dict]) -> list[dict]:
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
                        for i in range(pages[0] - 1, pages[0]):
                            writer.add_page(reader.pages[i])

                        with open(segmented_local_file, 'wb') as output_pdf:
                            writer.write(output_pdf)
                    raw_sample['file_name'] = segmented_local_file

        if "answerfile_name" in raw_sample and Path(raw_sample["answerfile_name"]).exists():
            raw_sample["ideal"] = Path(raw_sample["answerfile_name"]).read_text()
    return raw_samples
