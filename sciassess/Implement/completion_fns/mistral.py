import os
from pathlib import Path
from .base_completion_fn import BaseCompletionFn
import time
import traceback
import logging
logger = logging.getLogger(__name__)
from openai import OpenAI

from sciassess.Implement.utils import PROJECT_PATH

from evals.api import CompletionFn, CompletionResult
from sciassess.Implement.completion_fns.utils import call_without_throw
from typing import Any, Optional, Union
from .utils import load_from_cache, get_file_content
import logging
from typing import List, Dict
logger = logging.getLogger(__name__)

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


class MistralCompletionFn(CompletionFn):
    def __init__(
            self,
            port = 8000,
            max_len = 8192,
            model = "vllm",
            cache_dir = os.path.join(PROJECT_PATH, "SciAssess_library/tmp/pypdf"),
            **kwargs
    ):
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(
            api_key = "EMPTY",
            base_url = f"http://localhost:{port}/v1"
        )
        self.model = model
        self.max_len = max_len

    @call_without_throw
    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        attached_file_content = ""
        if "file_name" in kwargs and self.pdf_parser == 'pypdf':
            pdf_path = kwargs['file_name']
            attached_file_content = load_from_cache(self.cache_dir, pdf_path)
            if len(attached_file_content) == 0:
                attached_file_content = get_file_content(self.cache_dir, pdf_path)
            attached_file_content = "\nThe whole content of this patent is as follow:\n```\n" + attached_file_content + "\n```\n"
        
        messages[-1]['content'] += attached_file_content
        if messages[0]['role'] == 'system':
            messages[1]['content'] = messages[0]['content'] + '\n' + messages[1]['content']
            messages = messages[1:]
        result = SimpleCompletionResult(self.get_completions(messages, **kwargs))
        return result

    def get_completions(self, messages, **kwargs) -> str:
        try:
            messages[-1]['content'] = messages[-1]['content'][:int(self.max_len*2*0.9)]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                timeout=300
            )
            result = completion.choices[0].message.content
        except:
            logger.info(f"Error occurred: {traceback.format_exc()}. Retrying in 3 seconds...")
            time.sleep(3)
            messages[-1]['content'] = messages[-1]['content'][:self.max_len]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                timeout=300
            )
            result = completion.choices[0].message.content
        return result
