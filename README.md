# ConTempoR

 **ConTempoR** (Conversational Temporal Reasoning Benchmark), a systematic methodology and resource for constructing temporally consistent dialogues from static knowledge bases. As illustrated in Figure 1, we generate multi-turn interactions by applying three Contextual Dependency Types and rigorously injecting temporal logic via four novel Temporal Relation Assignment Strategies:

– S1: Pronoun-Based Coreference Reasoning. Tests coreference resolution within temporal contexts by substituting repeated entities with pronouns to simulate natural discourse.

– S2: Explicit Temporal Connective Reasoning. Evaluates the ability to follow chronological logic using explicit markers (e.g., “Meanwhile”).

– S3: Relative Temporal-Shift Reasoning. Demands arithmetic reasoning to calculate valid time scopes based on precise relative offsets (e.g., “Two decades later”).

– S4: Abstract Temporal-Shift Reasoning. Simulates vague memory by
replacing precise constraints with abstract offsets (e.g., “A few months later”),

 It reuses parts of the [TIQ](https://github.com/zhenjia2017/TIQ) codebase but extensively refactors and extends them in package. Evidence retrieval runs through [CLOCQ](https://github.com/PhilippChr/CLOCQ).

- Repository: <https://github.com/MrHoux/ConTempoR>

## Installation

```bash
git clone https://github.com/MrHoux/ConTempoR
cd ConTempoR

# Conda (recommended)
conda env create -f environment.yml
conda activate ConTempoR
pip install -e .

# Or with pip only
pip install -r requirements.txt
```

### Dependencies

ConTempoR makes use of [CLOCQ](https://github.com/PhilippChr/CLOCQ) for retrieving facts from WIKIDATA.
CLOCQ can be conveniently integrated via the [publicly available API](https://clocq.mpi-inf.mpg.de), using the client
from [the repo](https://github.com/PhilippChr/CLOCQ). CLOCQ must be initialized before using the client.

## Running the pipeline

To construct the benchmark requires following major steps.

#### 1. Retrieve year pages from Wikipedia

```bash
  bash scripts/pipeline.sh --year-page-retrieve config/config-ConTempoR.yml
```

#### 2. Construct pseudo-questions with GPT

```bash
  bash scripts/pipeline.sh --pseudoquestion-generate config/config-ConTempoR.yml
```

#### 3. Process questions with rephrasing and connecting

```bash
  bash scripts/pipeline.sh --question-rephrase config/config-ConTempoR.yml
```

## Data
You need the following data. You can download
from [here](https://qa.mpi-inf.mpg.de/faith/data_for_reproduce_faith.tar.gz):

- wikipedia_wikidata_mappings.pickle
- wikipedia_mappings.pickle
- wikidata_mappings.pickle
- types.pickle
- labels.pickle
- augmented_wikidata_mappings.pickle

## Benchmark

`benchmark/full.json`: aggregated all conversations with timestamp-normalised evidence.
- `benchmark/train.json` (1–350)
- `benchmark/dev.json` (351–450)
- `benchmark/test.json` (451–606)

## Credits & Guidelines

- ConTempoR builds upon [TIQ](https://github.com/zhenjia2017/TIQ) under permission and keeps the same CLOCQ dependency chain.
- Please follow CLOCQ’s licence and usage rules.

## Citation
This code is based on the method from:
```bibtex
@article{jia2024faithful,
  title={Faithful Temporal Question Answering over Heterogeneous Sources},
  author={Jia, Zhen and Christmann, Philipp and Weikum, Gerhard},
  journal={arXiv preprint arXiv:2402.15400},
  year={2024}
}
@inproceedings{christmann2022beyond,
  title={Beyond NED: Fast and Effective Search Space Reduction for Complex Question Answering over Knowledge Bases},
  author={Christmann, Philipp and Saha Roy, Rishiraj and Weikum, Gerhard},
  booktitle={Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  pages={172--180},
  year={2022}
}