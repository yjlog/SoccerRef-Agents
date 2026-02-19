# SoccerRef-Agents: Multi-Agent System for Automated Soccer Refereeing


**SoccerRef-Agents** is a holistic and explainable multi-agent decision-making framework for automated soccer refereeing. Unlike traditional black-box models, our system decomposes the officiating task into perception, retrieval, legal interpretation, and final adjudication, mimicking the collaborative workflow of a professional refereeing team.


## Key Features

* **⚽ SoccerRefBench:** A multimodal benchmark comprising **1,218 theoretical questions** (from CFA, NFHS, FHSAA exams) and **600 video judgment scenarios** (from SoccerNet-MVFoul).
* **📚 RefKnowledgeDB:** A vector-based knowledge base containing the digitized *Laws of the Game (2025/26)* and a curated *Classic Case Database* for precise, knowledge-driven reasoning.
* **🤖 Multi-Agent Architecture:** A collaborative system featuring specialized agents:
    * **Video Agent:** Perception and visual description.
    * **Rule Agent:** Legal interpretation via RAG.
    * **Case Agent:** Historical precedent retrieval via CBR.
    * **Context Agent:** Match background analysis.
    * **Chief Referee Agent:** Final adjudication and explanation generation.
* **🔍 Explainability:** Provides "glass-box" reasoning traces, grounding decisions in specific rule clauses and historical precedents.


## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yjlog/SoccerRef-Agents.git

```


2. **Install dependencies:**
It is recommended to use a virtual environment (Python 3.9+).
```bash
pip install -r requirements.txt

```


3. **Prepare Data:**
* The textual data and knowledge base are included in `Database/`.
* **For Video Data:** Due to copyright and size constraints, we provide the metadata in `Database/Video/video_600.json`. You must download the raw clips from [SoccerNet-MVFoul](https://github.com/SoccerNet/sn-mvfoul) and organize them locally.

## Citation

If you find this code or dataset useful for your research, please cite our paper:

```bibtex
@article{meng2026soccerrefagents,
  title={SoccerRef-Agents: Multi-Agent System for Automated Soccer Refereeing},
  author={Meng, Zi and Song, Wanli and Hu, Yi and Rao, Jiayuan and Chen, Gang},
  year={2026}
}

```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/yjlog/SoccerRef-Agents/blob/main/LICENSE) file for details.

## Authors

* **Zi Meng** (University of Michigan)
* **Wanli Song** (University of Michigan)
* **Yi Hu**, **Jiayuan Rao**, **Gang Chen** (Shanghai Jiao Tong University)

## Acknowledgment
We would like to express our gratitude to the following resources and communities that made this research possible:

* **[SoccerNet Team](https://www.soccer-net.org/):** For providing the *SoccerNet-MVFoul* dataset, which served as the foundation for our video judgment benchmark.
* **The International Football Association Board (IFAB):** For publishing the transparent *Laws of the Game*, enabling the construction of our logic knowledge base.
* **Open Source Community:** Special thanks to the developers of [ChromaDB](https://www.trychroma.com/), [OpenAI SDK](https://github.com/openai/openai-python), and [LangChain](https://www.langchain.com/) for their powerful tools.

## Contact

If you have any questions regarding the code, dataset, or the paper, please feel free to open an issue on this repository or contact the author directly:

**Zi Meng** (University of Michigan): [mengzi@umich.edu](mailto:mengzi@umich.edu)



