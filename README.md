<p align="center">
    <img src="assets/logo.png" width="150" style="margin-bottom: 0.2;"/>
<p>

<h5 align="center"> 🌸 "Just when the caterpillar thought the world was over, it became a butterfly." 🦋 </h2>

<h3 align="center"><a href="https://arxiv.org/abs/2409.18786">
Large Language Models Can Self-Improve in Long-context Reasoning</a></h3>
<!-- <h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2> -->

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2409.18786-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2409.18786)
[![hf_paper](https://img.shields.io/badge/%F0%9F%A4%97-Paper-FF6F61
)](https://huggingface.co/papers/2409.18786)
[![hf_model_data](https://img.shields.io/badge/%F0%9F%A4%97-Models&Datasets-48A9DC
)](https://huggingface.co/collections/Siheng99/sealong-67313e3b4edd034cb4a76cc5)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</h5>

## 📰 News
* **[2024.11.10]**  Release training and evaluation codes, models, and datasets for SEALONG.


## 🛠️ Requirements and Installation
**Basic Dependencies**:
* Python >= 3.10
* Pytorch >= 2.4.0
* CUDA Version >= 12.1

**Install required packages**:
```bash
git clone https://github.com/SihengLi99/SEALONG
pip install -r requirements.txt
```

## 🔑 Usage
**Model Usage**:
```python
import transformers
import torch

model_id = "Siheng99/Llama-3.1-8B-Instruct-SEALONG"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

**Data Usage**:
```python
from datasets import load_dataset
dataset = load_dataset("Siheng99/Llama-3.1-8B-Instruct-SEALONG-Dataset")
print(dataset)
print(dataset["train"][0])
```


## 📊 Evaluation
```bash
bash scripts/eval_longbench_qa.sh
```
Note: Set MODEL_NAME_OR_PATH to the desired target model.

## 🔥 Training
### Data Preparation

#### 1. Synthesizing Your Own Data
**Download MuSiQue**:
```bash
cd data
gdown 'https://drive.google.com/uc?export=download&id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h'
unzip musique_data_v1.0.zip -d musique && mv musique/data/* musique/ 
rm -r musique/data && rm musique_data_v1.0.zip
```

**Process MuSiQue**:
```bash
bash scripts/process_data.sh
```

**Synthesize Training Data**:
```bash
bash scripts/synthesize.sh
```

#### 2. Using Our Pre-synthesized Data
```python
from datasets import load_dataset
dataset = load_dataset("Siheng99/Llama-3.1-8B-Instruct-SEALONG-Dataset")
dataset.save_to_disk(/path/to/your/save_dir)
```

### Fine-tuning

Set MODEL_NAME_OR_PATH and DATASET in the scripts before fine-tuning.

**ORPO**:
```bash
# QLoRA
bash scripts/finetune_orpo_qlora_xtuner.sh
# Full-parameter
bash scripts/finetune_orpo_xtuner.sh
```

**SFT**:

You may also opt for SFT; however, our findings indicate that ORPO achieves superior performance (see Table 5 in our paper).
```bash
# QLoRA
bash scripts/finetune_sft_qlora_xtuner.sh
# Full-parameter
bash scripts/finetune_sft_xtuner.sh
```

In our experiments, we select QLoRA for memory efficiency, but we also test full parameter training. We observe that a learning rate of 5e-6 yields decent performance when using ORPO with full parameter training.

## 📑 Citation

If SEALONG is useful for your research or applications, please cite it with the following BibTeX:
```bibtex
@article{li2024survey,
      title={A Survey on the Honesty of Large Language Models},
      author={Siheng Li and Cheng Yang and Taiqiang Wu and Chufan Shi and Yuji Zhang and Xinyu Zhu and Zesen Cheng and Deng Cai and Mo Yu and Lemao Liu and Jie Zhou and Yujiu Yang and Ngai Wong and Xixin Wu and Wai Lam},
      year={2024},
      journal={arXiv preprint arXiv:2409.18786}
}
```

## 👍 Acknowledgement
We gratefully acknowledge the following projects that SEALONG builds upon:
* [**MuSiQue**](https://github.com/StonyBrookNLP/musique)
* [**XTuner**](https://github.com/InternLM/xtuner)
* [**DeepSpeed**](https://github.com/microsoft/DeepSpeed)
* [**transformers**](https://github.com/huggingface/transformers)
* [**datasets**](https://github.com/huggingface/datasets)
* [**peft**](https://github.com/huggingface/peft)
* [**orpo**](https://github.com/xfactlab/orpo)
