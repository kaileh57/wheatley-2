# Experiment 3: Dataset Mixing

## Overview
This experiment tests different mixing strategies for combining synthetic datasets to maximize model capabilities.

## Quick Start

1. **Create mixed datasets:**
   ```bash
   python data_mixer.py --strategy all --total_samples 100000
   ```

2. **Train models on mixed datasets:**
   ```bash
   python train_mixed_models.py --strategy equal_mix --model_size 500M
   python train_mixed_models.py --strategy instruction_heavy --model_size 500M
   # etc.
   ```

3. **Evaluate results:**
   ```bash
   python analyze_mixing_results.py
   ```

## Mixing Strategies

- **equal_mix**: 25% each dataset
- **instruction_heavy**: 40% OpenHermes, 30% Magpie, 20% Cosmopedia, 10% FineWeb
- **knowledge_heavy**: 40% Cosmopedia, 30% FineWeb, 20% OpenHermes, 10% Magpie
- **conversation_heavy**: 50% Magpie, 20% each others
- **quality_weighted**: Based on quality scores from Experiment 2
- **capability_balanced**: Optimized for diverse capabilities