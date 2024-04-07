from .base_completion_fn import BaseCompletionFn

class Qwen(BaseCompletionFn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add your initialization code here


    def get_completions(self, messages, **kwargs) -> str:
        # 接受一个messages列表，如[{"role": "user", "content": "你好"}]
        # 返回一个字符串
        return "Hello, world!"
