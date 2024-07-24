# Finetune-example

Here is a simple example of using our dataset to fine-tune a model. In this case, LLaMA-Factory is used to quickly fine-tune the Qwen2-7B-Instruct model.



## Installation

please refer to  https://github.com/hiyouga/LLaMA-Factory and https://huggingface.co/Qwen/Qwen2-7B-Instruct.



## Custom dataset

You need to convert the data into a specific format for fine-tuning, for example, into instruction fine-tuning data in the ShareGPT format. `SciAssess/finetune-example-config/sciassess_sft.json` is an example dataset.

(!!! Please note that the example dataset provided here is only a subset of the original dataset for a single sub-task. **Additionally, how to customize your fine-tuning dataset, how to handle context, how to parse PDFs, and the token length limit for each data entry are all open questions.**)



## Finetune 

Place the `sciassess_sft.json` file in the data folder within the `LLaMA-Factory` directory. Modify the `dataset_info.json` file inside the `LLaMA-Factory/data` folder by adding the following code snippet:

```json
"sciassess_sft":{
    "file_name":"sciassess_sft.json",
    "formatting":"sharegpt",
    "columns":{
      "messages":"input"
    },
    "tags":{
      "role_tag":"role",
      "content_tag":"content",
      "user_tag": "user",
      "system_tag": "system",
      "assistant_tag":"assistant"
      
    }
    }
```



Then,place the `qwen2-instruct-7B-lora_sft.yaml` to the `LLaMA-Factory/examples/train_lora` .(You can modify the 'yaml' to fix your config)



Use lora to finetune the model:

```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/qwen2-instruct-7B-lora_sft.yaml
```



## Merge the model

```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --adapter_name_or_path ./saves/qwen2-7B-Instruct/lora/sft \
    --template qwen \
    --finetuning_type lora \
    --export_dir  ./ckpt/qwen2-7B-Instruct-lora \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False
```



For more details, please refer to the official documentation of llama-factory and qwen2.

https://github.com/hiyouga/LLaMA-Factory

https://huggingface.co/Qwen/Qwen2-7B-Instruct