<h1 align="center">
    <p> DenseLoRA: Dense Low-Rank Adaptation of Large Language Models <br> [ACL 2025(main)]</p>
</h1>

## Quick Start and some tricks regarding finetuning with DenseLoRA
### Setup
Install dependencies
```bash
pip install -r requirements.txt
```

### Training(finetune.py)
An example could be as follows:
```bash
sh llama3_8B_denselora.sh 16 32 ./finetuned_result/llama3-8b/ 0 
```

### Inerence
An example could be as follows:
```bash
sh llama3_8B_denselora_eval.sh ./finetuned_result/llama3-87b/ 0
```

## Citation
If you find DenseLoRA useful, please consider giving a star and citation:
```bibtex
@Inproceedings{mu2025denselora,
  title={DenseLoRA: Dense Low-Rank Adaptation of Large Language Models},
  author={Lin Mu, Xiaoyu Wang, Li Ni, Yang Li, Zhize Wu, Peiquan Jin, Yiwen Zhang},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2025}
}
```

## Acknowledgement
This repo benefits from LLM-Adapters, DoRA. Thanks for their wonderful works.

