# FaVChat_final

A standalone runnable repository for the FaVChat reproduction built from the original `FaVChat` workspace.

## Included
- FaVChat GRPO entry: `src/r1-v/src/open_r1/favchat_grpo.py`
- FaVChat SFT entry: `src/r1-v/src/open_r1/sft_favchat.py`
- FaVChat trainer: `src/r1-v/src/open_r1/trainer/favchat_grpo_trainer.py`
- Hierarchical prompt-query modules: `src/r1-v/src/open_r1/favchat_components.py`
- Local `qwen_vl_utils` dependency: `src/qwen-vl-utils`
- Training scripts: `src/scripts/run_favchat_grpo.sh`, `src/scripts/run_favchat_sft.sh`

## Installation
```bash
cd FaVChat_final
bash setup.sh
```

## Data layout
Place training data under:
- `src/r1-v/FaVChat-data/FaVChat-170k.json`
- `src/r1-v/FaVChat-data/FaVChat-COT-170k.json`

Place referenced media files under `src/r1-v/FaVChat-data/` according to the `path` fields in the dataset JSON.

## Training
SFT:
```bash
bash src/scripts/run_favchat_sft.sh
```

GRPO:
```bash
bash src/scripts/run_favchat_grpo.sh
```

## Notes
- This repository is trimmed to the FaVChat-related training/evaluation codepath.
- The DE-GRPO implementation is in `src/r1-v/src/open_r1/trainer/favchat_grpo_trainer.py`.
- The hierarchical prompt-query encoder is in `src/r1-v/src/open_r1/favchat_components.py`.
