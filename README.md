# Medical Chat Summarization with Fine-Tuned BART (SOAP)

## Overview

This project fine-tunes a sequence-to-sequence language model to generate SOAP-style (Subjective, Objective, Assessment, Plan) medical summaries from doctor–patient chat conversations. 
The goal is to convert long, unstructured dialogues into concise, accurate, and readable clinical summaries, following the requirements of Task 03 (B).
The workflow includes baseline evaluation, supervised fine-tuning, quantitative evaluation, and qualitative analysis.


## Dataset

- **Source:** Medical Chat Summarization
- **Format:** CSV / Excel
- **Columns:**
  - `dialogue` – Medical chat
  - `soap` – Reference SOAP summary

### Splits
- Training: `medical_dialogue_train.csv`
- Validation: `medical_dialogue_validation.xlsx`
- Test: `medical_dialogue_test.xlsx`


## Model

- **Base Model:** `facebook/bart-base`
- **Type:** Encoder–Decoder Transformer
- **Why BART-BASE:**
  - Bart-large uses 12 layers in enc-dec, while bart-base uses only 6
  - Commonly used for summarization
  - Stable training with limited data
  - Well supported by Hugging Face tools

---
## Files and Folders


Data and outputs folder were too large so they were not added to the repository.
```
.
│
├── data/
│ ├── medical_dialogue_train.csv
│ ├── medical_dialogue_validation.xlsx
│ └── medical_dialogue_test.xlsx
│
├── outputs/
│ └── model/       #final model will be stored here after training.
│     └── checkpoint-27750/   
│
├── main/
│ ├── load.py # Dataset inspection
│ ├── base.py # Baseline (pre-trained BART)
│ ├── train.py # Fine-tuning script
│ └── eval.py # Evaluation (ROUGE + BERTScore)
│
├── requirements.txt
└── README.md
```


## Setup

- Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Dependencies

```bash
torch
transformers
datasets
evaluate
sentencepiece
pandas
accelerate
openpyxl
absl-py 
rouge_score 
nltk
bert_score
protobuf
sentencepiece
tiktoken
tokenizersß
```

## Model Information

### Architecture Selection

| Model | Type | Reason for Selection |
|-------|------|----------------------|
| `facebook/bart-base` | Encoder–Decoder (Seq2Seq Transformer) | Designed for summarization tasks<br>Strong performance on low-resource and structured generation tasks<br>Compatible with Hugging Face's Seq2SeqTrainer<br>Efficient enough to train on limited hardware (Mac MPS / CPU) |

### Baseline Evaluation 

Before fine-tuning, the pre-trained BART model was used directly:

```bash
python base.py
```
Result of [baseline] (https://docs.google.com/document/d/1M4ZUO0ohzazs2MHcSIoy1Wn66JoW20dUC-qQvOF88Y0/edit?usp=sharing). 
Before fine-tuning, the baseline model simply prints a summary identical to the input dialogue. This is beacause the untrained BART model:
  - Was trained on general English data, not medical dialogues
  - Does not understand medical terms yet


### Fine-Tuning

#### Preprocessing
- **Input**: `dialogue` (max 512 tokens)
- **Target**: `soap` (max 700 tokens)
- Padding and truncation applied
- Supervised Seq2Seq training

#### Training Configuration
| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 1 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Gradient checkpointing | Enabled |
| Evaluation | Per epoch |

```bash
python train.py
```
Challenges:
- Long dialogues causing truncation
- Small batch size (hardware limits)
- Preserving factual medical details

### Evaluation

The fine-tuned model was evaluated on the test set using lexical and semantic metrics.

#### Metrics
| Metric | Description |
|--------|-------------|
| **ROUGE** | Measures overlap with reference summaries |
| **BERTScore** | Semantic similarity |

```bash
python eval.py
```
## Results

The fine-tuned BART model was evaluated on the held-out test set using both lexical overlap and semantic similarity metrics. Here are the [results] (https://docs.google.com/document/d/1-_Msafv46d-SpfIsW156UmEK5zBmT1vNiaGYNP2Kza0/edit?usp=sharing)

### Quantitative Metrics

**ROUGE**
- ROUGE-1: **0.641** (64% unigram overlap - matching individual words)
- ROUGE-2: **0.394** (39% bigram overlap - matching word pairs)
- ROUGE-L: **0.476** (48% longest common sequence overlap)
- ROUGE-Lsum: **0.570**


The model's summaries share ~64% of individual words with the reference summaries. 
  

**BERTScore**
- Average Precision: **~0.91**
- Average Recall: **~0.91**
- Average F1: **~0.84**

These scores indicate strong overlap with reference summaries and high semantic similarity, showing that the model captures both structure and meaning effectively.


## Qualitative Analysis

### Successful Case

In the example shown during evaluation, the model:
- Accurate clinical data capture with correct numerical values
- Proper SOAP format structure
- Comprehensive objective findings
- Concise while preserving critical info

This demonstrates the model’s ability to condense long, complex medical dialogues into clinically meaningful summaries.

### Failure Modes

Some limitations were also observed:
- Missing explicit "no significant past medical history" statement
- Assessment section lacks differential diagnosis details
- Plan less specific about specialist consultation rationale

Most errors occurred when important information was spread across multiple dialogue turns or implied rather than explicitly stated.


## Discussion

Compared to the baseline pre-trained model, the fine-tuned model shows a clear improvement in:
- SOAP structure consistency
- Medical content accuracy
- Overall coherence and conciseness

The combination of ROUGE and BERTScore provides a balanced evaluation, capturing both surface-level overlap and deeper semantic similarity. Despite hardware constraints and limited batch size, the model demonstrates strong performance on Bengali medical chat summarization.


### Conclusion

Fine-tuning BART on domain-specific medical chats significantly improves SOAP summarization quality compared to the baseline. 
The final model produces more structured, relevant, and clinically meaningful summaries.
