from evals.api import CompletionFn, CompletionResult
from sciassess.Implement.completion_fns.utils import call_without_throw
from typing import Any, Optional, Union
from .utils import extract_text
import logging
from typing import List, Dict
logger = logging.getLogger(__name__)

class SimpleCompletionResult(CompletionResult):
    def __init__(self, response: str) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()] if self.response else ["Unknown"]

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
        if "file_name" in kwargs:
            attached_file_content = "\nThe file is as follows:\n\n" + "".join(extract_text(kwargs["file_name"], self.pdf_parser))
            kwargs.pop('file_name')
        else:
            attached_file_content = ""
        messages[-1]['content'] += attached_file_content
        return self.get_completions(messages=messages, **kwargs)


    def get_completions(self, messages: List[Dict], **kwargs: Any) -> str:
        raise NotImplementedError