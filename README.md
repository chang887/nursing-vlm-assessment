# Automated Procedural Analysis via Video-Language Models for Scalable Training of Nursing Skills

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/your-paper-id) 
[![Dataset](https://img.shields.io/badge/Dataset-NurViD-blue)](https://drive.google.com/drive/folders/1ibALEtA6xqnmuKYHMTFrvCUucNzP8oQW?usp=sharing)
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/chang887/models)

The first VLM-based framework for automated procedural assessment in nursing education, enabling scalable training through curriculum-inspired hierarchical evaluation.

![Framework Overview](assets/framework_overview.png)

## Overview

This framework addresses critical challenges in nursing education by automating procedural assessment and feedback generation. Our system mimics human skill acquisition through a progressive learning approach, advancing from high-level action recognition to fine-grained temporal analysis and sophisticated procedural reasoning.

### Key Capabilities

- **Error Diagnosis**: Automatically identifies missing or incorrect subactions in nursing procedures
- **Explainable Feedback**: Generates natural language explanations for procedural errors
- **Standardized Evaluation**: Provides consistent, objective assessment across institutions
- **Multi-procedure Support**: Handles diverse nursing skills including venipuncture, wound care, and catheterization

## Installation

```bash
git clone https://github.com/your-repo/nursing-vlm-framework.git
cd nursing-vlm-framework
bash scripts/install.sh
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 32GB+ GPU memory (recommended for 7B model)

## Dataset Preparation

### NurViD Dataset

Download the NurViD dataset from [Google Drive](https://drive.google.com/drive/folders/1ibALEtA6xqnmuKYHMTFrvCUucNzP8oQW?usp=sharing):

```bash
# Create data directory structure
mkdir -p data/NurViD/{videos,annotations,processed}

# Place downloaded files:
# - Raw videos in data/NurViD/videos/
# - Annotations in data/NurViD/annotations/
# - Processed clips in data/NurViD/processed/
```

The dataset includes:
- **1.5K video instances** spanning 51 nursing procedures
- **177 granular action primitives** with temporal annotations
- **Expert-validated** procedural sequences

### Data Processing Pipeline

```bash
# Generate fine-grained temporal annotations
python scripts/data_processing/generate_annotations.py \
    --input_dir data/NurViD/videos \
    --output_dir data/NurViD/annotations

# Create masked prediction datasets  
python scripts/data_processing/create_masked_events.py \
    --annotations_dir data/NurViD/annotations \
    --output_dir data/NurViD/processed/masked_events

# Generate sequence correction datasets
python scripts/data_processing/create_shuffled_sequences.py \
    --annotations_dir data/NurViD/annotations \
    --output_dir data/NurViD/processed/shuffled_sequences
```

## Model Architecture

Our framework builds upon **Qwen2.5-VL** with a curriculum-inspired multistage training strategy:

### Training Stages

1. **Stage 1**: Coarse-grained Procedure Recognition
2. **Stage 2**: Fine-grained Dense Event Understanding  
3. **Stage 3**: Causal Reasoning via Missing Event Prediction
4. **Stage 4**: Chronological Reasoning via Order Correction

## Quick Start

### Inference

```python
from src.inference import NursingVLMPredictor

# Load trained model
predictor = NursingVLMPredictor.from_pretrained("chang887/nursing-vlm-7b-s4")

# Analyze nursing procedure video
results = predictor.analyze_procedure("path/to/nursing_video.mp4")

print(f"Procedure: {results['procedure']}")
print(f"Temporal Segments: {results['segments']}")
print(f"Detected Errors: {results['errors']}")
print(f"Feedback: {results['feedback']}")
```

### Demo Application

Launch the interactive demo:

```bash
python src/serve/app.py --model_path chang887/nursing-vlm-7b-s4
```

Navigate to `http://localhost:7860` to access the web interface.

## Training

### Prerequisites

```bash
# Install training dependencies
pip install deepspeed wandb

# Configure DeepSpeed
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Stage-by-Stage Training

#### Stage 1: Procedure Recognition

```bash
bash scripts/train/stage1_procedure_recognition.sh \
    --model_id Qwen/Qwen2.5-VL-7B \
    --data_path data/NurViD/stage1_procedures.json \
    --output_dir checkpoints/stage1 \
    --num_train_epochs 3
```

#### Stage 2: Dense Event Understanding

```bash
bash scripts/train/stage2_dense_captioning.sh \
    --model_id checkpoints/stage1 \
    --data_path data/NurViD/stage2_dense_events.json \
    --output_dir checkpoints/stage2 \
    --num_train_epochs 2
```

#### Stage 3: Missing Event Prediction

```bash
bash scripts/train/stage3_missing_events.sh \
    --model_id checkpoints/stage2 \
    --data_path data/NurViD/stage3_masked_events.json \
    --output_dir checkpoints/stage3 \
    --num_train_epochs 2
```

#### Stage 4: Sequence Order Correction

```bash
bash scripts/train/stage4_order_correction.sh \
    --model_id checkpoints/stage3 \
    --data_path data/NurViD/stage4_shuffled_sequences.json \
    --output_dir checkpoints/stage4 \
    --num_train_epochs 2
```

### Training Configuration

<details>
<summary>Training Arguments</summary>

- `--model_id`: Base model or checkpoint path
- `--data_path`: Training data JSON file
- `--output_dir`: Checkpoint output directory
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate (stage-specific)
- `--lora_enable`: Enable LoRA fine-tuning
- `--lora_rank`: LoRA rank (default: 32)
- `--deepspeed`: DeepSpeed configuration file
- `--bf16`: Enable bfloat16 training

</details>

## Task Examples

### Task 1: Procedure Identification

**Input**: Untrimmed nursing video  
**Output**: Procedure name and temporal segments

```json
{
  "procedure": "Perineal Care",
  "segments": [
    [10.78, 18.51],
    [187.61, 200.15],
    [200.51, 253.9],
    [316.71, 366.75]
  ]
}
```

### Task 2: Fine-grained Action Segmentation

**Input**: Nursing procedure video  
**Output**: Detailed temporal segments with captions

```json
[
  {
    "start_time": 0,
    "end_time": 3,
    "caption": "The nurse approaches the patient, who is lying on a hospital bed, and exchanges information with a colleague."
  },
  {
    "start_time": 4,
    "end_time": 7,
    "caption": "The nurse adjusts the patient's head and shoulder area, ensuring alignment and comfort."
  }
]
```

### Task 3: Missing Event Detection

**Input**: Video with masked segments  
**Output**: Reasoning and predicted missing actions

```json
{
  "reasoning": "The video shows a nurse smoothing a sheet over a patient from 3s to 5s. Before this action, the nurse likely needs to prepare the patient and bed.",
  "predicted_event": "The nurse adjusts the bed or the patient's position to ensure they are ready for the sheet to be smoothed over them.",
  "time_range": [0, 2]
}
```

### Task 4: Sequence Order Correction

**Input**: Video with shuffled action sequence  
**Output**: Correctness assessment and proper ordering

```json
{
  "is_reasonable": false,
  "out_of_place": [
    {
      "segment": "25–28",
      "reason": "Tilting the patient's upper body should occur before lifting and adjusting the lower body."
    }
  ],
  "correct_order": [
    {
      "start": 0,
      "end": 5,
      "caption": "The nurse approaches the patient and reviews initial positioning."
    }
  ],
  "reasoning": [
    "The nurse begins by approaching the patient and reviewing positioning.",
    "Tilting the upper body should occur before adjusting the lower body."
  ]
}
```

## Performance Results

| Task | Metric | Performance | Improvement |
|------|--------|-------------|-------------|
| Procedure ID | Top-1 Accuracy | 31.4% | +25.5% vs base |
| Dense Captioning | F1@IoU≥0.5 | 0.352 | +51.7% vs base |
| Missing Detection | F1 Score | 0.620 | +55.4% vs base |
| Order Correction | Hit@0.5s | 0.1257 | +102% vs base |

## Model Weights

Pre-trained models are available on Hugging Face:

- [Stage 1 Model](https://huggingface.co/chang887/nursing-vlm-7b-s1): Procedure recognition
- [Stage 2 Model](https://huggingface.co/chang887/nursing-vlm-7b-s2): Dense captioning  
- [Stage 3 Model](https://huggingface.co/chang887/nursing-vlm-7b-s3): Causal reasoning
- [Stage 4 Model](https://huggingface.co/chang887/nursing-vlm-7b-s4): Chronological reasoning

## Applications

### Educational Integration

- **Simulation Labs**: Automated assessment of student performance
- **Competency Evaluation**: Standardized skill validation
- **Remediation**: Personalized feedback for skill improvement
- **Faculty Support**: Reduced instructor workload

### Clinical Training

- **Continuing Education**: Ongoing skill assessment for practicing nurses
- **Certification**: Objective competency validation
- **Quality Assurance**: Consistent procedural adherence monitoring

## Contributing

We welcome contributions to improve the framework:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{nursing_vlm_2025,
  title={Automated Procedural Analysis via Video-Language Models for Scalable Training of Nursing Skills},
  author={Chang, S. and Liu, D. and Tian, R. and Swartzell, K.L. and Klingler, S.L. and Nagle, A.M. and Kong, N.},
  journal={IISE Transactions},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We build upon the following repositories:
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B): Foundation VLM architecture
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): Efficient fine-tuning framework
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): Distributed training optimization

## Contact

For questions and collaboration opportunities:
- **Email**: chang887@purdue.edu
- **Issues**: [GitHub Issues](https://github.com/your-repo/nursing-vlm-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/nursing-vlm-framework/discussions)