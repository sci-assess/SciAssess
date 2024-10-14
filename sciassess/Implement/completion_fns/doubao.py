import os
from pathlib import Path
from .base_completion_fn import BaseCompletionFn
import time
import traceback
import logging
logger = logging.getLogger(__name__)
from volcenginesdkarkruntime import Ark

from sciassess.Implement.utils import PROJECT_PATH


# load api key
access_key = "<access_key>"
secret_key = "<secret_key>"
client = Ark(ak=access_key, sk=secret_key)

model = "ep-20240527074836-gsn56"

class DoubaoCompletionFn(BaseCompletionFn):
    def __init__(
            self,
            cache_dir = os.path.join(PROJECT_PATH, "SciAssess_library/tmp/pypdf"),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def get_completions(self, messages, **kwargs) -> str:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                timeout=300
            )
            result = completion.choices[0].message.content
        except:
            logger.info(f"Error occurred: {traceback.format_exc()}. Retrying in 60 seconds...")
            time.sleep(60)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                timeout=300
            )
            result = completion.choices[0].message.content
        return result
