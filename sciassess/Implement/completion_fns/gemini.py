import os
import time
import logging
from typing import Any, Optional, Union
from evals.api import CompletionFn, CompletionResult

from sciassess.Implement.completion_fns.utils import call_without_throw
from vertexai.generative_models import GenerativeModel, Part
from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types
    
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "<credentials_file_path>"

multimodal_model = GenerativeModel(model_name="gemini-1.5-pro")
safety_settings={
    gapic_content_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: gapic_content_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    gapic_content_types.HarmCategory.HARM_CATEGORY_HARASSMENT: gapic_content_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    gapic_content_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: gapic_content_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    gapic_content_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: gapic_content_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    gapic_content_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED: gapic_content_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
}

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


class GeminiCompletionFn(CompletionFn):
    def __init__(
            self,
            **kwargs
    ):
        pass

    @call_without_throw
    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        contents = []

        if 'file_name' in kwargs:
            with open(kwargs["file_name"], 'rb') as f:
                pdf_data = Part.from_data(f.read(), mime_type="application/pdf")
            contents.append(pdf_data)

        messages_str = ""
        for msg in messages:
            messages_str += f"{msg['role']}: {msg['content']}\n"
        contents.append(messages_str)

        for attempt in range(10):
            try:
                response = multimodal_model.generate_content(contents, safety_settings=safety_settings)
                result = SimpleCompletionResult(response.candidates[0].content.parts[0].text)
                return result
            except Exception as e:
                if attempt < 10 - 1:
                    logging.info(f"failed: {e}. Retrying...")
                    time.sleep(10)
                else:
                    raise
