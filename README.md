# SciAssess: A Benchmark for Evaluating Large Language Models in Scientific Literature Analysis

### Version: 1.0.0

SciAssess is a comprehensive benchmark designed to evaluate the proficiency of Large Language Models (LLMs) in scientific literature analysis. It focuses on assessing LLMs' abilities in memorization, comprehension, and analysis within the context of scientific literature, covering a wide range of scientific fields such as general chemistry, organic materials, and alloy materials. SciAssess provides a rigorous and thorough assessment of LLMs, supporting the ongoing development of LLM applications in scientific literature analysis.

For more details, please refer to our paper: [SciAssess: Benchmarking LLM Proficiency in Scientific Literature Analysis](https://arxiv.org/abs/2403.01976).

### Domains and Tasks
| Domain                                                                                                                                                                                                          | Task                                                                                                        | Ability | \# Questions | Context    | Question Type       | Metric          | Modality    |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|---------|--------------|------------|---------------------|-----------------|-------------|
| Fundamental Science                                                                                                                                                             | MMLU (science)                                                                            | L1      | 2,091        |            | Multiple Choice     | Accuracy        | Text only   |
|                                                                                                                                                                                                                 | CMMLU (science)                                                                           | L1      | 1,700        |            | Multiple Choice     | Accuracy        | Text only   |
|                                                                                                                                                                                                                 | Xiezhi-Ch (science)                                                                      | L1      | 2,882        |            | Multiple Choice     | Accuracy        | Text only   |
|                                                                                                                                                                                                                 | Xiezhi-En (science)                                                                     | L1      | 2,882        |            | Multiple Choice     | Accuracy        | Text only   |
| Alloy Materials                                                                                                                                                                 | Alloy Chart QA                                                                                              | L2      | 15           | ✔️ | Multiple Choice     | Accuracy        | Chart       |
|                                                                                                                                                                                                                 | Composition Extraction                                                                                      | L2      | 244          | ✔️ | Table Extraction    | Table Accuracy  | Table       |
|                                                                                                                                                                                                                 | Temperature Extraction                                                                                      | L2      | 207          | ✔️ | Multiple Choice     | Accuracy        | Text only   |
|                                                                                                                                                                                                                 | Sample Differentiation                                                                                      | L3      | 237          | ✔️ | Multiple Choice     | Accuracy        | Text only   |
|                                                                                                                                                                                                                 | Treatment Sequence                                                                                          | L3      | 102          | ✔️ | True/False          | Accuracy        | Text only   |
|                                                                                                                                                                                       Biomedicine               | Biology Chart QA                                                                                            | L2      | 99           | ✔️ | Multiple Choice     | Accuracy        | Chart       |
|                                                                                                                                                                                                                 | Chemical Entities Recognition                                                             | L2      | 997          |            | Text Extraction     | Recall          | Text only   |
|                                                                                                                                                                                                                 | Disease Entities Recognition                                                              | L2      | 997          |            | Text Extraction     | Recall          | Text only   |
|                                                                                                                                                                                                                 | Compound Disease Recognition                                                              | L3      | 997          |            | Text Extraction     | Recall          | Text only   |
|                                                                                                                                                                                                                 | Gene Disease Function                                                                     | L3      | 236          |            | Text Extraction     | Recall          | Text only   |
|                                                                                                                                                                                                                 | Gene Disease Regulation                                                                   | L3      | 240          |            | Text Extraction     | Recall          | Text only   |
| Drug Discovery                                                                                                                                                                  | Affinity Extraction                                                                                         | L2      | 40           | ✔️ | Table Extraction    | Table Accuracy  | Mol., Table |
|                                                                                                                                                                                                                 | Drug Chart QA                                                                                               | L2      | 15           | ✔️ | Multiple Choice     | Accuracy        | Chart       |
|                                                                                                                                                                                                                 | Tag to Molecule                                                                                             | L2      | 50           | ✔️ | Molecule Generation | Mol. Similarity | Mol.        |
|                                                                                                                                                                                                                 | Markush to Molecule                                                                                         | L3      | 37           |            | Molecule Generation | Mol. Similarity | Mol.        |
|                                                                                                                                                                                                                 | Molecule in Document                                                                                        | L3      | 50           | ✔️ | True/False          | Accuracy        | Mol.        |
|                                                                                                                                                                                                                 | Reaction QA                                                                                                 | L3      | 95           | ✔️ | Multiple Choice     | Accuracy        | Reaction    |
|                                                                                                                                                                                                                 | Drug Target Identification                                                                                  | L3      | 40           | ✔️ | Text Extraction     | Recall          | Text only   |
|                                                                                                                                                                         Organic Materials                       | Electrolyte Table QA                                                                                        | L2      | 100          | ✔️ | Multiple Choice     | Accuracy        | Table       |
|                                                                                                                                                                                                                 | OLED Property Extraction                                                                                    | L2      | 13           | ✔️ | Table Extraction    | Table Accuracy  | Mol.,Table  |
|                                                                                                                                                                                                                 | Polymer Chart QA                                                                                            | L2      | 15           | ✔️ | Multiple Choice     | Accuracy        | Chart       |
|                                                                                                                                                                                                                 | Polymer Composition QA                                                                                      | L2      | 109          | ✔️ | Multiple Choice     | Accuracy        | Text only   |
|                                                                                                                                                                                                                 | Polymer Property Extraction                                                                                 | L2      | 109          | ✔️ | Table Extraction    | Table Accuracy  | Table       |
|                                                                                                                                                                                                                 | Solubility Extraction                                                                                       | L2      | 100          | ✔️ | Table Extraction    | Table Accuracy  | Table       |
|                                                                                                                                                                                                                 | Reaction Mechanism QA                                                                                       | L3      | 22           | ✔️ | Multiple Choice     | Accuracy        | Reaction    |



## Performance
### Table1: Overall Performance
| Domain              | Task                                                           | ICL    | GPT-4o                    | GPT-4                     | GPT-3.5                   | Moonshot                  | Claude3                   | Doubao                    | Gemini                    | Llama3                  | DeepSeek                | Qwen2                   | Command R+   |
|:-------------------:|:--------------------------------------------------------------:|:------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------------:|
| Fundamental Science | MMLU (science)                                                 | 0-shot | 0.839                     | 0.783                     | 0.629                     | 0.774                     | 0.795                     | 0.720                     | 0.799                     | 0.766                   | 0.737                   |  0.782  | 0.647                         |
|                     |                                                                | 3-shot |  0.846  | 0.769                     | 0.614                     | 0.774                     | 0.771                     | 0.712                     | 0.790                     | 0.757                   | 0.738                   |  0.789  | 0.643                         |
|                     | CMMLU (science)                                                | 0-shot | 0.785                     | 0.644                     | 0.438                     | 0.723                     | 0.643                     |  0.841  | 0.731                     | 0.651                   | 0.769                   |  0.870  | 0.448                         |
|                     |                                                                | 3-shot | 0.785                     | 0.646                     | 0.432                     | 0.728                     | 0.631                     |  0.833  | 0.736                     | 0.658                   | 0.768                   |  0.867  | 0.455                         |
|                     | Xiezhi-Ch (science)                                            | 0-shot |  0.736  | 0.724                     | 0.696                     | 0.734                     | 0.731                     | 0.720                     | 0.716                     | 0.731                   |  0.748  | 0.746                   | 0.683                         |
|                     |                                                                | 3-shot |  0.736  | 0.708                     | 0.690                     | 0.732                     | 0.706                     | 0.706                     | 0.723                     | 0.736                   | 0.726                   |  0.745  | 0.672                         |
|                     | Xiezhi-En (science)                                            | 0-shot |  0.701  | 0.683                     | 0.644                     | 0.677                     | 0.673                     | 0.667                     | 0.652                     | 0.687                   | 0.685                   |  0.692  | 0.634                         |
|                     |                                                                | 3-shot |  0.699  | 0.670                     | 0.641                     | 0.679                     | 0.658                     | 0.650                     | 0.654                     | 0.683                   | 0.665                   |  0.697  | 0.632                         |
|    Alloy Materials                 | Alloy Chart QA                                                 | 0-shot | 0.533                     | 0.600                     | 0.333                     | 0.333                     | 0.400                     | 0.467                     |  0.667  |  0.467  | 0.333                   | 0.400                   | 0.200                         |
|                     | Composition Extraction                                         | 0-shot | 0.484                     | 0.458                     | 0.112                     | 0.127                     |  0.495  | 0.304                     | 0.239                     | 0.212                   | 0.389                   |  0.423  | 0.128                         |
|                     | Temperature Extraction                                         | 0-shot | 0.884                     | 0.855                     | 0.729                     |  0.889  | 0.865                     | 0.700                     | 0.841                     | 0.604                   | 0.754                   |  0.797  | 0.546                         |
|                     | Sample Differentiation                                         | 0-shot | 0.511                     | 0.591                     | 0.169                     |  0.679  | 0.586                     | 0.316                     | 0.658                     | 0.376                   |  0.616  | 0.557                   | 0.228                         |
|                     | Treatment Sequence                                             | 0-shot | 0.745                     | 0.725                     | 0.461                     |  0.755  | 0.745                     | 0.745                     | 0.696                     | 0.539                   |  0.686  | 0.657                   | 0.588                         |
|        Biomedicine           |   Biology Chart QA               | 0-shot | 0.580                     | 0.480                     | 0.390                     | 0.545                     | 0.505                     | 0.480                     |  0.616  | 0.520                   |  0.545  | 0.515                   | 0.535                         |
|                     |                                    Chemical Entities Recognition                            | 0-shot | 0.454                     | 0.665                     | 0.540                     | 0.201                     | 0.844                     |  0.911  | 0.678                     | 0.400                   | 0.536                   | 0.832                   |  0.850        |
|                     |     | 3-shot |  0.916  | 0.898                     | 0.912                     | 0.912                     | 0.898                     | 0.900                     | 0.858                     | 0.855                   |  0.911  | 0.905                   | 0.871                         |
|                     |                       Disease Entities Recognition                                         | 0-shot | 0.279                     |  0.765  | 0.153                     | 0.000                     | 0.653                     | 0.675                     | 0.437                     | 0.526                   | 0.331                   |  0.722  | 0.258                         |
|                     |      | 3-shot | 0.822                     | 0.849                     |  0.879  | 0.785                     | 0.782                     | 0.811                     | 0.807                     | 0.787                   | 0.825                   |  0.826  | 0.647                         |
|                     |                              Compound Disease Recognition                                  | 0-shot | 0.755                     | 0.786                     | 0.733                     | 0.770                     |  0.788  | 0.771                     | 0.733                     |  0.794  | 0.757                   |  0.794  | 0.764                         |
|                     |      | 3-shot | 0.743                     | 0.750                     | 0.715                     |  0.773  | 0.763                     | 0.719                     | 0.719                     |  0.785  | 0.716                   | 0.753                   | 0.715                         |
|                     |                                      Gene Disease Function                          | 0-shot | 0.931                     |  0.974  | 0.864                     | 0.771                     | 0.944                     | 0.779                     | 0.954                     |  0.996  | 0.819                   | 0.930                   | 0.884                         |
|                     |             | 3-shot |  0.945  | 0.927                     | 0.896                     | 0.845                     | 0.931                     | 0.772                     | 0.868                     | 0.876                   | 0.830                   | 0.814                   |  0.888        |
|                     |                                  Gene Disease Regulation                              | 0-shot |  0.949  | 0.914                     | 0.832                     | 0.944                     | 0.939                     | 0.910                     | 0.856                     |  0.971  | 0.952                   | 0.963                   | 0.936                         |
|                     |         | 3-shot | 0.939                     | 0.926                     | 0.917                     |  0.957  | 0.951                     | 0.912                     | 0.886                     |  0.958  | 0.943                   | 0.953                   | 0.936                         |
|      Drug Discovery               | Affinity Extraction                                            | 0-shot | 0.072                     | 0.042                     | 0.025                     | 0.040                     |  0.097  | 0.050                     | 0.040                     | 0.064                   | 0.017                   |  0.075  | 0.043                         |
|                     | Drug Chart QA                                                  | 0-shot | 0.333                     | 0.400                     | 0.067                     | 0.400                     | 0.200                     |  0.533  |  0.533  | 0.400                   | 0.400                   | 0.400                   |  0.533        |
|                     | Tag to Molecule                                                | 0-shot | 0.040                     | 0.022                     | 0.000                     | 0.016                     | 0.035                     | 0.094                     |  0.169  |  0.034  | 0.014                   | 0.000                   | 0.031                         |
|                     |   Markush to Molecule            | 0-shot | 0.634                     | 0.632                     | 0.429                     | 0.462                     |  0.644  | 0.217                     | 0.218                     | 0.478                   |  0.543  | 0.358                   | 0.332                         |
|                     |                                                                | 3-shot | 0.642                     | 0.654                     | 0.431                     | 0.504                     |  0.675  | 0.239                     | 0.526                     |  0.491  | 0.470                   | 0.379                   | 0.376                         |
|                     | Molecule in Document                                           | 0-shot | 0.580                     |  0.700  | 0.500                     | 0.460                     | 0.480                     | 0.560                     | 0.640                     |  0.680  | 0.460                   | 0.460                   | 0.460                         |
|                     | Reaction QA                                                    | 0-shot |  0.705  | 0.674                     | 0.442                     | 0.253                     | 0.663                     | 0.442                     | 0.305                     |  0.611  | 0.368                   | 0.442                   | 0.316                         |
|                     | Drug Target Identification                                     | 0-shot | 0.721                     | 0.791                     | 0.526                     | 0.607                     |  0.794  | 0.622                     | 0.768                     | 0.600                   |  0.687  | 0.410                   | 0.485                         |
|      Organic Materials               | Electrolyte Table QA                                           | 0-shot |  0.940  | 0.790                     | 0.370                     | 0.670                     | 0.870                     | 0.710                     | 0.880                     | 0.460                   |  0.720  | 0.620                   | 0.450                         |
|                     | OLED Property Extraction                                       | 0-shot | 0.336                     | 0.406                     | 0.201                     | 0.037                     |  0.477  | 0.259                     | 0.093                     | 0.263                   | 0.292                   |  0.392  | 0.234                         |
|                     | Polymer Chart QA                                               | 0-shot | 0.800                     | 0.667                     | 0.400                     | 0.800                     | 0.467                     |  0.867  | 0.800                     | 0.867                   | 0.733                   |  0.933  | 0.800                         |
|                     | Polymer Composition QA                                         | 0-shot |  0.945  |  0.945  | 0.853                     | 0.844                     | 0.881                     | 0.927                     | 0.927                     | 0.734                   | 0.881                   |  0.936  | 0.679                         |
|                     | Polymer Property Extraction                                    | 0-shot | 0.692                     | 0.681                     | 0.329                     |  0.705  | 0.629                     | 0.514                     | 0.606                     | 0.536                   |  0.652  | 0.636                   | 0.171                         |
|                     | Solubility Extraction                                          | 0-shot |  0.479  | 0.440                     | 0.410                     | 0.363                     | 0.426                     | 0.371                     | 0.397                     | 0.399                   |  0.432  | 0.400                   | 0.351                         |
|                     | Reaction Mechanism QA                                          | 0-shot | 0.545                     | 0.636                     | 0.455                     | 0.545                     | 0.455                     | 0.636                     |  0.727  | 0.500                   | 0.545                   |  0.591  |  0.591        |
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

**0.9.2** (2024-04-06) Optimize the metric of multiple choice questions.

**0.9.3** (2024-04-07) Merge mmlu college chemistry and high school chemistry.

Remove abstract2title and research_question_extraction due to uncertainty of model grading.

**1.0.0** (2024-04-08) Official version released

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
