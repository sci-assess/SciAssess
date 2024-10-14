# SciAssess: Benchmarking LLM Proficiency in Scientific Literature Analysis
Recent breakthroughs in Large Language Models (LLMs) have revolutionized scientific literature analysis. 
However, existing benchmarks fail to adequately evaluate the proficiency of LLMs in this domain, particularly in scenarios requiring higher-level abilities beyond mere memorization and the handling of multimodal data.
In response to this gap, we introduce SciAssess, a benchmark specifically designed for the comprehensive evaluation of LLMs in scientific literature analysis. 
It aims to thoroughly assess the efficacy of LLMs by evaluating their capabilities in Memorization (L1), Comprehension (L2), and Analysis & Reasoning (L3). 
It encompasses a variety of tasks drawn from diverse scientific fields, including biology, chemistry, material, and medicine.
To ensure the reliability of SciAssess, rigorous quality control measures have been implemented, ensuring accuracy, anonymization, and compliance with copyright standards. 
SciAssess evaluates 11 LLMs, highlighting their strengths and areas for improvement. 
We hope this evaluation supports the ongoing development of LLM applications in scientific literature analysis.

## Installation

To use SciAssess, first clone the repository.

Install the required dependencies:

```bash
conda create -n evals python=3.10
conda activate evals
pip install -e .
```

Additional Considerations:

In some task evaluations, we use models from `sentence-transformers` on Hugging Face. Please ensure that you can connect to Hugging Face. If you are unable to connect, you might consider manually downloading the corresponding models and updating the model import path in `./sciassess/Implement/utils/metrics.py` to reflect the location where you have placed the models.

## Dataset

Due to copyright restrictions, we are unable to directly distribute the original PDF of the article. You will need to download the corresponding PDF according to the instructions in README and store it in SciAssess_library/pdfs.

All articles involved in this evaluation are listed in [doi.txt](doi.txt). You need to download the corresponding PDFs according to the DOIs and store them in SciAssess_library/pdfs.

Each PDF should be named as doi.pdf, with '/' in the DOI replaced by '_', e.g., an article with DOI 10.1002/adfm.202008332 should be named as 10.1002_adfm.202008332.pdf and placed in SciAssess_library/pdfs.

Some articles' supporting information is also evaluated. These articles' DOIs are listed in [si_doi.txt](si_doi.txt). You need to download the corresponding PDFs and store them in SciAssess_library/pdfs, named as doi_si.pdf.


## Usage


Remember to export your OpenAI API base and key as an environment variable:

```bash
export OPENAI_BASE_URL=your_openai_base_url
export OPENAI_API_KEY=your_openai_api_key
```


```bash
bash run_sciassess.sh model_name
```
Replace `model_name` with the name of your model (default: `gpt3.5`).
If you want to add CoT, add `export USE_COT=1` to `run_sciassess.sh` before `scievalset` commands.

---

*model list* 🔥
- o1-preview
- gpt4o
- gpt4
- gpt3.5
- moonshot
- claude3
- doubao
- gemini

## Deploy your own model
Example for Qwen2.5-72B-Instrct:

### deploy model with vllm
```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --dtype auto --disable-custom-all-reduce --enforce-eager --served-model-name vllm --tensor-parallel-size 8
```

### register model on evals
Add model in `sciassess/Registry/completion_fns/vllm.yaml`

Change `port` and `max_len` with your own parameters
```yaml
Qwen2.5-72B-Instruct:
  class: sciassess.Implement.completion_fns.vllm:VllmCompletionFn
  args:
      port: 8000
      max_len: 131072
```

### eval model on SciAssess-v3
model_name is decided by model info in `sciassess/Registry/completion_fns/vllm.yaml`
```bash
bash run_sciassess.sh Qwen2.5-72B-Instruct
```