import logging
from typing import List,Dict,Any,Union
from evals.api import CompletionResult
from .base_completion_fn import BaseCompletionFn
from openai import OpenAI
import time
from sciassess.Implement.completion_fns.utils import call_without_throw
from .utils import extract_text

openai_api_key="EMPTY"
openai_api_base = "http://localhost:8000/v1"
logger = logging.getLogger(__name__)

client= OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

class Qwen2CompletionResult(CompletionResult):
    def __init__(self,response:str) -> None:
        self.response = response

    def get_completions(self) -> List[str]:
        return [self.response.strip()] if self.response else ["Unknown"]
    
class Qwen2CompletionFn(BaseCompletionFn):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.model="Qwen2-7B-Instruct"

    def get_completions(self, messages, **kwargs) -> str:
        chat_response= client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        response = chat_response.choices[0].message.content
        return  Qwen2CompletionResult(response)        

    @call_without_throw
    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        if "file_name" in kwargs:
            attached_file_content = "\nThe file is as follows:\n\n" + "".join(extract_text(kwargs["file_name"], self.pdf_parser))
            #wyc-temp-change
            attached_file_content = attached_file_content[:2048]   
            kwargs.pop('file_name')
        else:
            attached_file_content = ""


         
        messages[-1]['content'] += attached_file_content
        return self.get_completions(messages=messages, **kwargs) 


        
    
