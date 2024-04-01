# SciAssess: A Benchmark for Evaluating Large Language Models in Scientific Literature Analysis

### Version: 0.9.0

SciAssess is a comprehensive benchmark designed to evaluate the proficiency of Large Language Models (LLMs) in scientific literature analysis. It focuses on assessing LLMs' abilities in memorization, comprehension, and analysis within the context of scientific literature, covering a wide range of scientific fields such as general chemistry, organic materials, and alloy materials. SciAssess provides a rigorous and thorough assessment of LLMs, supporting the ongoing development of LLM applications in scientific literature analysis.

For more details, please refer to our paper: [SciAssess: Benchmarking LLM Proficiency in Scientific Literature Analysis](https://arxiv.org/abs/2403.01976).

## Benchmark
### Evaluated Abilities
- **L1 (Memorization)**: The model's ability to accurately answer common factual questions in science autonomously.
- **L2 (Comprehension)**: The ability to precisely identify and extract key information and facts within a given text.
- **L3 (Analysis and Reasoning)**: The model's capability to amalgamate extracted information with its existing knowledge base for logical reasoning and analysis.

### Domains and Tasks
| Domain            | Task                         | Ability | # Questions | Question Type        | Multimodal Content  |
|-------------------|------------------------------|---------|-------------|----------------------|---------------------|
| General Chemistry | MMLU High-School Chemistry   | L1      | 22          | Multiple Choice      |                     |
|                   | MMLU College Chemistry       | L1      | 8           | Multiple Choice      |                     |
|                   | Abstract2Title               | L2      | 254         | Open-ended Generation|                     |
|                   | Question Extraction          | L2      | 19          | Open-ended Generation|                     |
|                   | Balancing Equations          | L3      | 100         | Constrained Generation|                    |
| Alloy Materials   | Composition Extraction       | L2      | 55          | Table Extraction     | Table               |
|                   | Target Extraction            | L2      | 50          | Multiple Choice      |                     |
|                   | Treatment Sequence           | L2      | 25          | True/False           |                     |
|                   | Alloy ChartQA                | L2      | 15          | Multiple Choice      | Chart               |
|                   | Sample Differentiation       | L3      | 50          | Multiple Choice      |                     |
| Organic Materials | Electrolyte Solubility Data Extraction | L2 | 8 | Table Extraction | Table           |
|                   | Electrolyte Table QA         | L2      | 48          | Multiple Choice      | Table               |
|                   | Reaction Mechanism QA        | L2      | 22          | Multiple Choice      | Molecule            |
|                   | Polymer Property Extraction  | L2      | 15          | Table Extraction     | Table               |
|                   | Polymer Composition Extraction | L2    | 15          | Table Extraction     |                     |
|                   | OLED Property Extraction     | L2      | 13          | Table Extraction     | Molecule, Table     |
|                   | Polymer ChartQA              | L2      | 15          | Table Extraction     | Chart               |
| Drug Discovery    | Affinity Data Extraction     | L2      | 15          | Table Extraction     | Molecule, Table     |
|                   | Tag to Molecule              | L2      | 41          | Constrained Generation | Molecule          |
|                   | Target Extraction            | L2      | 15          | Constrained Generation |                    |
|                   | Drug ChartQA                 | L2      | 15          | Multiple Choice      | Chart               |
|                   | Reaction QA                  | L2      | 15          | Multiple Choice      | Reaction            |
|                   | Molecule in Document         | L3      | 45          | True/False           | Molecule            |
|                   | Markush to Molecule          | L3      | 9           | Constrained Generation | Molecule          |
| Biology           | MedMCQA                      | L1      | 100         | Multiple Choice      |                     |
|                   | CompDisease Recognition      | L1      | 500         | Text Extraction      |                     |
|                   | GeneDisease Text Mining      | L2      | 75          | Text Comprehension   |                     |
|                   | Biology ChartQA              | L2      | 15          | Multiple Choice      | Chart               |

## Installation

To use SciAssess, first clone the repository:

```bash
git clone https://github.com/sci-assess/SciAssess.git
cd SciAssess
```

Install the required dependencies:

```bash
pip install -e .
```

## Dataset

Due to copyright restrictions, we are unable to directly distribute the original PDF of the article. You will need to download the corresponding PDF according to the instructions in README and store it in SciAssess_library/pdfs.

All articles involved in this evaluation are listed in [doi.txt](doi.txt). You need to download the corresponding PDFs according to the DOIs and store them in SciAssess_library/pdfs.

Each PDF should be named as doi.pdf, with '/' in the DOI replaced by '_', e.g., an article with DOI 10.1002/adfm.202008332 should be named as 10.1002_adfm.202008332.pdf and placed in SciAssess_library/pdfs.

Some articles' supporting information is also evaluated. These articles' DOIs are listed in [si_doi.txt](si_doi.txt). You need to download the corresponding PDFs and store them in SciAssess_library/pdfs, named as doi_si.pdf.

## Usage

If you want to evaluate your own model, you need to configure your model's registration information and implementation in sciassess/Registry/completion_fns and sciassess/Implement/completion_fns, respectively. See [openai/evals:completion-fns.md](https://github.com/openai/evals/blob/main/docs/completion-fns.md) for configuration instructions.

Note that most evaluations depend on the article PDFs, so you may need to process the input PDFs within your model's method. The PDF file path will be passed in the  `__call__` function through kwargs['file_name'], and you need to handle this parameter and process the PDF in the  `__call__` function. See [openai_with_pdf.py](sciassess/Implement/completion_fns/openai_with_pdf.py) for an example based on PyPDF and GPT.

After completing the model configuration, run the following command to evaluate your model:

```bash
bash run_sciassess.sh your_model_name
```

Replace `your_model_name` with the name of your model (default: `gpt3.5`).

Remember to export your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

## Version Information
**0.9.0** (2024-03-17) Beta version first released

**0.9.1** (2024-03-28) Fix critical bugs. Now the code is executable.

## Contributing

We welcome contributions to the SciAssess benchmark. If you have any suggestions or improvements, please feel free to open an issue or create a pull request.

## Citation

If you use SciAssess in your research, please cite our paper:

```
@misc{cai2024sciassess,
      title={SciAssess: Benchmarking LLM Proficiency in Scientific Literature Analysis}, 
      author={Hengxing Cai and Xiaochen Cai and Junhan Chang and Sihang Li and Lin Yao and Changxin Wang and Zhifeng Gao and Yongge Li and Mujie Lin and Shuwen Yang and Jiankun Wang and Yuqi Yin and Yaqi Li and Linfeng Zhang and Guolin Ke},
      year={2024},
      eprint={2403.01976},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
