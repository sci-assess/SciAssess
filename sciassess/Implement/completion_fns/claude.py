# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to generate a message with Anthropic Claude (on demand).
"""
import os
from pathlib import Path
import json
from evals.api import CompletionFn, CompletionResult
from evals.record import record_sampling
from sciassess.Implement.utils import PROJECT_PATH
from sciassess.Implement.completion_fns.utils import call_without_throw
from typing import Any, Optional, Union
from .utils import load_from_cache, get_file_content

import logging
logger = logging.getLogger(__name__)
import boto3

session = boto3.Session(
    region_name='<region_name>',
    aws_access_key_id = "<access_key_id>",
    aws_secret_access_key = "<secret_access_key>"
)
bedrock_runtime = session.client(service_name='bedrock-runtime')
model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'

class ClaudeCompletionResult(CompletionResult):
    def __init__(self, response: str) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        if not self.response:
            return ["Unknown"]
        self.response = self.response.strip()
        if "Answer:" in self.response:
            self.response = self.response.split("Answer:")[1].strip()
        return [self.response]

class AwsClaude(CompletionFn):

    def __init__(
            self,
            instructions: Optional[str] = "You are a helpful assistant.",
            n_ctx: Optional[int] = None,
            extra_options: Optional[dict] = None,
            pdf_parser: Optional[str] = 'pypdf',
            cache_dir = os.path.join(PROJECT_PATH, "SciAssess_library/tmp/pypdf"),
            **kwargs
    ):
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.instructions = instructions
        self.n_ctx = n_ctx
        self.extra_options = extra_options if extra_options else {}
        self.pdf_parser = pdf_parser

    @call_without_throw
    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any):
        if isinstance(prompt, str):
            system_prompt = self.instructions
            messages = [{"role": "user", "content": prompt}]
        elif prompt[0]['role'] == 'system':
            system_prompt = prompt[0]['content']
            messages = prompt[1:]
        else:
            system_prompt = self.instructions
            messages = prompt

        attached_file_content = ""
        if "file_name" in kwargs and self.pdf_parser == 'pypdf':
            pdf_path = kwargs['file_name']
            attached_file_content = load_from_cache(self.cache_dir, pdf_path)
            if len(attached_file_content) == 0:
                attached_file_content = get_file_content(self.cache_dir, pdf_path)
            attached_file_content = "\nThe whole content of this patent is as follow:\n```\n" + attached_file_content + "\n```\n"
        
        messages[-1]['content'] += attached_file_content
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "system": system_prompt,
                "messages": messages,
                "max_tokens": 150_000,
            }
        )
        response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get('body').read())
        return ClaudeCompletionResult(response_body['content'][0]['text'])



