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

class BaseCompletionFn(CompletionFn):
    def __init__(
            self,
            instructions: Optional[str] = "You are a helpful assistant.",
            n_ctx: Optional[int] = None,
            extra_options: Optional[dict] = None,
            pdf_parser: Optional[str] = 'pypdf',
            **kwargs
    ):
        self.instructions = instructions
        self.n_ctx = n_ctx
        self.extra_options = extra_options if extra_options else {}
        self.pdf_parser = pdf_parser

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
        result = SimpleCompletionResult(self.get_completions(messages, **kwargs))
        return result

    def get_completions(self, messages: List[Dict], **kwargs: Any) -> str:
        raise NotImplementedError