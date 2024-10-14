import logging
import random
from typing import Any, Optional, Union

import os
import traceback
from openai import OpenAI

from evals.api import CompletionFn, CompletionResult
from evals.base import CompletionFnSpec
from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from evals.completion_fns.openai import OpenAIChatCompletionResult, OpenAICompletionResult
from .utils import extract_text, ErrorCompletionResult, call_without_throw, cache_to_disk
from openai import AzureOpenAI
import openai
from pathlib import Path
from sciassess.Implement.utils import PROJECT_PATH
from .utils import load_from_cache, get_file_content

key_4 = [
    {"api_base": "<api_base>", "api_key": "<api_key>", },
]

key_35 = [
    {"api_base": "<api_base>", "api_key": "<api_key>", },
]

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


class OpenAIChatCompletionFnAzure(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = None,
        n_ctx: Optional[int] = None,
        api_version = '2023-10-01-preview',
        pdf_parser: Optional[str] = 'pypdf',
        extra_options: Optional[dict] = {},
        cache_dir = os.path.join(PROJECT_PATH, "SciAssess_library/tmp/pypdf"),
        **kwargs,
    ):
        self.model = model
        self.key_pool = key_4 if model.startswith('gpt-4') else key_35
        self.n_ctx = n_ctx
        self.extra_options = extra_options
        self.pdf_parser = pdf_parser
        self.api_version = api_version
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @call_without_throw
    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAIChatCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = ChatCompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreateChatPrompt = prompt.to_formatted_prompt()

        attached_file_content = ""
        if "file_name" in kwargs:
            pdf_path = kwargs['file_name']
            attached_file_content = load_from_cache(self.cache_dir, pdf_path)
            if len(attached_file_content) == 0:
                attached_file_content = get_file_content(self.cache_dir, pdf_path)
            attached_file_content = "\nThe whole content of this patent is as follow:\n```\n" + attached_file_content + "\n```\n"
        
        openai_create_prompt[-1]["content"] += attached_file_content
        if self.model.startswith('gpt-35'):
            openai_create_prompt[-1]['content'] = openai_create_prompt[-1]['content'][:int(16000*2*0.9)]

        choose_api = random.choice(self.key_pool)
        api_base = choose_api["api_base"]
        api_key = choose_api['api_key']
        client = AzureOpenAI(
            azure_endpoint=api_base,
            api_key=api_key,
            api_version=self.api_version
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=openai_create_prompt,
            timeout=300,
            temperature=0,
            stream=False
        )
        result = SimpleCompletionResult(response.choices[0].message.content)
        return result
