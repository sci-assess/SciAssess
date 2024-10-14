import os
import random
from pathlib import Path
from .base_completion_fn import BaseCompletionFn
import time
import traceback
import logging
logger = logging.getLogger(__name__)
from openai import OpenAI

from sciassess.Implement.utils import PROJECT_PATH


class VllmCompletionFn(BaseCompletionFn):
    def __init__(
            self,
            port = 8000,
            max_len = 8192,
            model = "vllm",
            cache_dir = os.path.join(PROJECT_PATH, "SciAssess_library/tmp/pypdf"),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.model = model
        self.max_len = max_len
        self.port = port

    def get_completions(self, messages, **kwargs) -> str:
        if isinstance(self.port, list):  
            selected_port = random.choice(self.port)
        else:
            selected_port = self.port
        self.client = OpenAI(
            api_key = "EMPTY",
            base_url = f"http://localhost:{selected_port}/v1"
        )
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
