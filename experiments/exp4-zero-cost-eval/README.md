# Experiment 4: Zero-Cost Evaluation Framework

## Overview
Comprehensive LLM evaluation framework that requires zero API calls and correlates with expensive metrics.

## Quick Start

1. **Test the framework:**
   ```bash
   python evaluator.py
   ```

2. **Evaluate a custom model:**
   ```python
   from evaluator import ZeroCostEvaluator
   evaluator = ZeroCostEvaluator()
   results = evaluator.evaluate_model("path/to/model")
   ```

## Features

- **Statistical Metrics**: Perplexity, diversity, quality proxies
- **Proxy Tasks**: Arithmetic, pattern completion, instruction following
- **Composite Scoring**: Weighted combination of all metrics
- **No API Costs**: Completely offline evaluation
- **Fast Evaluation**: 1000x faster than GPT-4 based evaluation

## Metrics

### Core Metrics
- Perplexity on evaluation texts
- N-gram diversity (1-4 grams)
- Vocabulary diversity
- Grammar and readability scores

### Proxy Tasks
- Arithmetic (addition, subtraction, multiplication)
- Pattern completion (number and letter sequences)
- Instruction following (simple commands)
- Factual knowledge (basic facts)
- Text completion (common phrases)

### Composite Scores
- Quality Score (40% weight)
- Task Performance (40% weight)  
- Diversity Score (20% weight)
- Overall Score (weighted combination)