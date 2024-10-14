import os
import time
import logging
from pathlib import Path
from openai import OpenAI
from typing import Any, Optional, Union
from evals.api import CompletionFn, CompletionResult

from sciassess.Implement.completion_fns.utils import call_without_throw
from sciassess.Implement.utils import PROJECT_PATH

# load api key
api_key = '<api_key>'
base_url = "https://api.moonshot.cn/v1"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

model = "moonshot-v1-128k"

class SimpleCompletionResult(CompletionResult):
    def __init__(self, response: str) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        if not self.response:
            return ["Unknown"]
        self.response = self.response.strip()
        if "Answer:" in self.response:
            self.response = self.response.split("Answer:")[1].strip()
        return [self.response]


class MoonshotCompletionFn(CompletionFn):
    def __init__(
            self,
            cache_dir = os.path.join(PROJECT_PATH, "SciAssess_library/tmp/moonshot"),
            **kwargs
    ):
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @call_without_throw
    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        if 'file_name' in kwargs:
            pdf_path = kwargs['file_name']
            file_content = self.load_from_cache(pdf_path)
            if file_content is None:
                file_content = self.get_file_content(pdf_path)
            if messages[0]['role'] == 'system':
                messages.insert(1, {"role": "system", "content": file_content})
            else:
                messages.insert(0, {"role": "system", "content": file_content})
        
        for attempt in range(10):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                result = SimpleCompletionResult(completion.choices[0].message.content)
                return result
            except Exception as e:
                if attempt < 10 - 1:
                    logging.info(f"failed: {e}. Retrying...")
                    time.sleep(10)
                else:
                    raise

    def load_from_cache(self, pdf_path):
        cache_path = os.path.join(self.cache_dir, f"{os.path.basename(pdf_path)}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return f.read()
        else:
            return None

    def get_file_content(self, pdf_path):
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        file_object = client.files.create(file=Path(pdf_path), purpose="file-extract")
        if file_object.status != 'ok':
            raise Exception(f"Failed to extract text from file {pdf_path}: {file_object.status}")
        file_content = client.files.content(file_id=file_object.id).text
        client.files.delete(file_id=file_object.id)
        with open(os.path.join(self.cache_dir, f"{os.path.basename(pdf_path)}.txt"), 'w') as f:
            f.write(file_content)
        return file_content