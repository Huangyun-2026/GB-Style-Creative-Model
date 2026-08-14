# GB-Style-Creative-Model
# 【GB‑Style Creative Assistant】(AI‑Powered Writing Aid)
**Tech Stack**: Qwen (4‑bit quantization), LoRA/DPO fine‑tuning, Gradio, Python

**Project Description**: Independently developed an AI‑powered writing assistant for female‑oriented creators. It addresses pain points including stylistic homogenization, logical inconsistency, and limited lexical diversity within specific GB‑style writing scenarios.

**Core Contributions**:
- Designed and enforced rigorous data screening and filtering rules (removal of misogynistic terms, establishment of GB‑style writing standards) to build high‑quality datasets for model fine‑tuning.
- Performed fine‑tuning and DPO training based on the Qwen model with a 4‑bit quantization strategy, achieving viable stylized generation under constrained hardware resources.
- Implemented a dual‑path fault‑tolerant generation architecture consisting of a primary model and a fallback model. Custom functional modules such as logic validation and neologism replacement were developed to improve the stability and practicality of generated texts.
- Built the web application with Gradio and deployed it on cloud servers, forming a sustainable product loop for ongoing usage and user feedback.

#  GB‑Style Writing Assistant
> A GB (Female‑dominant, Male‑submissive) novel‑writing tool fine‑tuned on Qwen2.5‑7B

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

##  Project Overview

**GB‑Style Writing Assistant** is an AI‑powered writing tool dedicated to GB (Female‑dominant, Male‑submissive) fiction generation. Built upon Alibaba Qwen2.5‑7B‑Instruct, the model adopts SFT (Supervised Fine‑Tuning), QLoRA and DPO (Direct Preference Optimization) training pipelines to reliably produce narratives consistent with GB literary tropes.

### Core Features

-  **GB‑Style Specialization**: Fine‑tuned on GB‑genre corpus, supporting custom vocabulary unique to female‑dominant fictional universes.
-  **Real‑time Generation Feedback**: Visual progress indicator during text generation.
- **Optional DeepSeek Logical Validation**: Integrate DeepSeek API to evaluate logical consistency and trigger auto‑rewrite.
-  **Multi‑turn Continuation**: Continue existing stories with custom plot guidance.
- **Targeted Paragraph Rewrite**: Select any segment and apply custom rewriting instructions.
-  **One‑click Copy**: Quickly export generated text.

---

##  Technical Architecture

### Overall Pipeline
```
Raw novel corpus
    ↓
Data cleaning & structuring (clean_data.py)
    ↓
GB‑specific vocabulary replacement (replace.py)
    ↓
Tokenizer vocabulary expansion (expand_vocab.py)
    ↓
QLoRA fine‑tuning
    ↓
DPO preference optimization
    ↓
Gradio Web UI (app.py)
```

### Tech Stack

| Component | Selection | Notes |
|---|---|---|
| **Base Model** | Qwen2.5‑7B‑Instruct | Alibaba Qwen 7B parameter instruction‑tuned LLM |
| **Fine‑tuning Paradigm** | QLoRA + DPO | 4‑bit quantized LoRA + Direct Preference Optimization |
| **Inference Framework** | Transformers + PEFT | Hugging Face ecosystem |
| **Web UI** | Gradio | Rapid interactive web interface |
| **Logic Validation** | DeepSeek API (optional) | External consistency evaluation service |
| **Deployment Target** | AutoDL / Local GPU Server | GPU‑accelerated inference |

### Project Structure
```
├── app.py                          # Gradio web application entry
├── clean_data.py                   # Raw data cleaning & structuring
├── replace.py                      # GB‑style vocabulary substitution
├── expand_vocab.py                 # Tokenizer vocabulary expansion
├── run_replacement.py              # Vocabulary replacement executor
├── train_full_4bit.py              # QLoRA fine‑tuning script
├── merge_qlora.py                  # Merge LoRA adapters into base model
├── incremental_train.py            # Incremental fine‑tuning script
├── merge_dpo_offload.py            # Merge DPO‑optimized model weights
├── requirements.txt                # Python dependency manifest
├── LICENSE                         # Apache‑2.0 license file
├── README.md                       # Project documentation
└── docs/
    └── training_log.md             # Full training run log
```

---

##  Quick Start

### Hardware Requirements

| Hardware | Minimum | Recommended |
|---|---|---|
| **GPU VRAM** | 16 GB | A100‑40GB |
| **System RAM** | 32 GB | 64 GB+ |
| **Disk Storage** | 50 GB | 100 GB+ |
| **CUDA** | 11.8+ | 12.4+ |

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:
```
gradio>=4.0.0
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
peft>=0.5.0
datasets>=2.10.0
requests>=2.28.0
sentencepiece>=0.1.99
protobuf>=3.20.0
bitsandbytes>=0.46.1
trl>=0.8.0
```

### Launch Web Service
```bash
cd /root/autodl‑tmp/model_training
python app.py --server_port=7861
```
Visit `http://<your‑ip>:7861` in browser.

---

##  Usage Guide

### Initial Story Generation
1. **Story Background**: Era and world setting (e.g. "Alternate ancient realm ruled by a female sovereign")
2. **Story Setup**: Character relationships and world‑building details
3. **Generation Requirements**: Tone, plot direction and emotional atmosphere
4. **Opening Passage**: Opening text of your story
5. **Target Word Count**: Desired output length
6. Click **Generate** to start creation.

### Story Continuation
Click **Continue Writing**. Input optional plot directions in the expandable panel; the model will extend the existing narrative.

### Paragraph Rewrite
1. Switch to **Rewrite** tab
2. Paste source text
3. Highlight unsatisfactory segments
4. Specify rewrite instructions
5. Click **Rewrite**

> Note on custom vocabulary:
> Post‑generation substitution rules are defined inside `GB_REPLACEMENTS` in `replace.py`. These are custom invented terms for this fictional universe and remain untranslated in code for stylistic consistency.

##  Detailed Training Workflow

### Data Preparation
1. **Raw corpus**: Load `.txt` / `.docx` files from local dataset directory.
2. **Data cleaning**: `clean_data.py` removes noise, splits chapters and slices text chunks.
3. **Lexicon substitution**: `replace.py` applies GB‑specific vocabulary transformation.
4. **Tokenizer expansion**: `expand_vocab.py` injects custom tokens into model vocabulary.

### Training Stages

#### Phase 1: QLoRA Supervised Fine‑Tuning
```bash
python train_full_4bit.py --epochs 1 --batch_size 1 --learning_rate 2e-4
```
- 4‑bit quantization for memory saving
- LoRA rank=64, alpha=128
- Training dataset: 4,322 GB‑genre SFT samples

#### Phase 2: DPO Direct Preference Optimization
```bash
python train_dpo.py
```
- Preference pairs generated via DeepSeek API
- Reinforce preference for GB‑style narrative patterns
- Beta=0.1, learning rate = 5e‑7

#### Phase3: Model Weight Merging
```bash
python merge_qlora.py
```
Merge LoRA adapters with base weights to produce a standalone deployable model.

### Training Metrics

| Stage | Train Loss | Validation Loss | Remarks |
|---|---|---|---|
| QLoRA SFT | 2.454 → 1.748 | 3.154 | 3 training cycles |
| DPO Optimization | 0.589 | 0.401 | 1 training cycle |

---

##  Configuration

### DeepSeek API (Optional)
Edit `DEEPSEEK_API_KEY` inside `app.py`:
```python
DEEPSEEK_API_KEY = "sk‑your‑deepseek‑api‑key"
```

### Model Path
Modify `MODEL_PATH` variable to point to your merged model checkpoint:
```python
MODEL_PATH = "/root/autodl‑tmp/model_training/output/dpo_merged_model_offload"
```

### Gradio Port
Specify listening port on startup:
python app.py --server_port=7861

## Generation Example

### Input
```
Story Background: Alternate ancient realm ruled by female emperors; men hold lower social status.
Story Setup: A world blending imperial court and martial‑arts realms. Women may serve in government; men commonly act as attendants.
Requirements: Emphasize female dominance and tense emotional interplay.
Opening Passage: Night falls. Candlelight flickers inside the imperial study. She lays down memorials and gazes toward the male kneeling on the floor.
Target Word Count: 800
```

### Output (Excerpt)
> Night falls. Candlelight flickers inside the imperial study. She lays down memorials and gazes toward the male kneeling on the floor. Her gaze lingers on his trembling nape, tangled emotions swirling within her. “Rise.” Her voice is low, languid, carrying unyielding authority. At those words, his body stiffens. He slowly lifts his head. Panic and bewilderment flash across his features, soon overshadowed by fear. The empress is well‑known for her stern authority; none dare defy her commands…

**Word count**: ~800｜**GB‑style compliant**: ✅｜**Reversed‑role plot**: ✅ None

---

##  Contributing

Contributions, issues and pull‑requests are welcome.

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing‑feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to your branch (`git push origin feature/amazing‑feature`)
5. Open a Pull Request

---

## License

This project is licensed under the [Apache 2.0](LICENSE) License.

---

##  Acknowledgements

- [Qwen](https://github.com/QwenLM/Qwen) — Alibaba open‑source large‑language model
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — Training & inference toolkit
- [Gradio](https://gradio.app/) — Interactive web UI framework
- [DeepSeek](https://deepseek.com/) — Logical‑validation API service

## Contact

Open an Issue or send email for feedback and suggestions.

** Star this repository if you find the project helpful!**
