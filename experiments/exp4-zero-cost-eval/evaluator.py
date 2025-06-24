#!/usr/bin/env python3
"""
Zero-Cost LLM Evaluation Framework
Comprehensive evaluation without expensive API calls
"""

import os
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    AutoTokenizer
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import textstat
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation framework"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    max_eval_samples: int = 1000
    generation_samples: int = 100
    cache_dir: str = "./cache/evaluation"
    enable_caching: bool = True
    seed: int = 42


class StatisticalMetrics:
    """Statistical metrics that don't require external models"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
    
    def calculate_perplexity(self, model: GPT2LMHeadModel, texts: List[str]) -> float:
        """Calculate perplexity on a set of texts"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        with torch.no_grad():
            for text in tqdm(texts[:self.config.max_eval_samples], desc="Computing perplexity"):
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                    padding=True
                ).to(self.config.device)
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                
                # Only count non-padding tokens
                attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
                num_tokens = attention_mask.sum().item()
                
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def calculate_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate various diversity metrics"""
        
        diversity_scores = {}
        
        # N-gram diversity
        for n in [1, 2, 3, 4]:
            all_ngrams = []
            for text in texts[:self.config.max_eval_samples]:
                tokens = text.lower().split()
                ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                all_ngrams.extend(ngrams)
            
            if all_ngrams:
                unique_ngrams = len(set(all_ngrams))
                total_ngrams = len(all_ngrams)
                diversity_scores[f'distinct_{n}'] = unique_ngrams / total_ngrams
        
        # Vocabulary diversity
        all_tokens = []
        for text in texts[:self.config.max_eval_samples]:
            tokens = [w.lower() for w in text.split() if w.isalnum()]
            all_tokens.extend(tokens)
        
        if all_tokens:
            unique_tokens = len(set(all_tokens))
            total_tokens = len(all_tokens)
            diversity_scores['vocab_diversity'] = unique_tokens / total_tokens
        
        # Entropy-based diversity
        token_counts = Counter(all_tokens)
        total_count = sum(token_counts.values())
        
        if total_count > 0:
            entropy = -sum(
                (count / total_count) * np.log2(count / total_count)
                for count in token_counts.values()
            )
            diversity_scores['entropy'] = entropy
        
        return diversity_scores
    
    def calculate_quality_proxies(self, texts: List[str]) -> Dict[str, float]:
        """Calculate quality proxy metrics"""
        
        quality_metrics = {}
        
        # Readability scores
        readability_scores = []
        sentence_variations = []
        coherence_scores = []
        
        for text in texts[:self.config.max_eval_samples]:
            # Readability
            try:
                flesch = textstat.flesch_reading_ease(text)
                if not np.isnan(flesch):
                    readability_scores.append(flesch)
            except:
                pass
            
            # Sentence length variation
            sentences = sent_tokenize(text)
            if len(sentences) > 1:
                lengths = [len(s.split()) for s in sentences if s.strip()]
                if lengths:
                    sentence_variations.append(np.std(lengths))
            
            # Coherence proxy (consecutive sentence similarity)
            if len(sentences) > 1:
                similarities = []
                for i in range(len(sentences) - 1):
                    sim = self._simple_sentence_similarity(sentences[i], sentences[i+1])
                    similarities.append(sim)
                if similarities:
                    coherence_scores.append(np.mean(similarities))
        
        quality_metrics['avg_readability'] = np.mean(readability_scores) if readability_scores else 0
        quality_metrics['sentence_variation'] = np.mean(sentence_variations) if sentence_variations else 0
        quality_metrics['coherence_proxy'] = np.mean(coherence_scores) if coherence_scores else 0
        
        # Grammar indicators
        grammar_scores = []
        for text in texts[:min(100, len(texts))]:  # Limit for performance
            score = self._simple_grammar_score(text)
            grammar_scores.append(score)
        
        quality_metrics['grammar_score'] = np.mean(grammar_scores) if grammar_scores else 0
        
        return quality_metrics
    
    def _simple_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Simple sentence similarity based on word overlap"""
        words1 = set(w.lower() for w in sent1.split() if w.isalnum() and w.lower() not in self.stop_words)
        words2 = set(w.lower() for w in sent2.split() if w.isalnum() and w.lower() not in self.stop_words)
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _simple_grammar_score(self, text: str) -> float:
        """Simple grammar scoring based on patterns"""
        score = 0.0
        
        # Check for capitalization
        sentences = sent_tokenize(text)
        if sentences:
            capitalized = sum(1 for s in sentences if s and s[0].isupper())
            score += 0.3 * (capitalized / len(sentences))
        
        # Check for proper punctuation
        punct_ratio = sum(1 for c in text if c in '.!?;:,') / len(text) if text else 0
        score += 0.2 * min(punct_ratio * 20, 1.0)  # Scale appropriately
        
        # Check for reasonable sentence lengths
        if sentences:
            lengths = [len(s.split()) for s in sentences]
            avg_length = np.mean(lengths)
            # Optimal sentence length is around 15-20 words
            length_score = 1.0 - abs(avg_length - 17.5) / 17.5
            score += 0.3 * max(0, length_score)
        
        # Check for varied vocabulary
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += 0.2 * unique_ratio
        
        return min(score, 1.0)


class ProxyTaskSuite:
    """Proxy tasks that correlate with model capabilities"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.tasks = self._create_tasks()
    
    def _create_tasks(self) -> Dict[str, List[Dict]]:
        """Create proxy task suites"""
        
        tasks = {
            'arithmetic': self._create_arithmetic_tasks(),
            'pattern_completion': self._create_pattern_tasks(),
            'instruction_following': self._create_instruction_tasks(),
            'factual_knowledge': self._create_knowledge_tasks(),
            'text_completion': self._create_completion_tasks()
        }
        
        return tasks
    
    def _create_arithmetic_tasks(self, n: int = 100) -> List[Dict]:
        """Create simple arithmetic tasks"""
        tasks = []
        
        for _ in range(n):
            op = random.choice(['+', '-', '*'])
            if op == '*':
                a, b = random.randint(2, 12), random.randint(2, 12)
            else:
                a, b = random.randint(10, 99), random.randint(10, 99)
            
            if op == '+':
                answer = a + b
            elif op == '-':
                answer = a - b
            else:
                answer = a * b
            
            tasks.append({
                'prompt': f"What is {a} {op} {b}?",
                'expected': str(answer),
                'checker': lambda response, ans=answer: str(ans) in response.replace(',', ''),
                'category': 'arithmetic'
            })
        
        return tasks
    
    def _create_pattern_tasks(self, n: int = 50) -> List[Dict]:
        """Create pattern completion tasks"""
        tasks = []
        
        # Number patterns
        for _ in range(n // 2):
            start = random.randint(1, 10)
            step = random.randint(2, 5)
            sequence = [start + i * step for i in range(4)]
            
            tasks.append({
                'prompt': f"Complete the pattern: {', '.join(map(str, sequence[:3]))}, __",
                'expected': str(sequence[3]),
                'checker': lambda response, ans=sequence[3]: str(ans) in response,
                'category': 'pattern'
            })
        
        return tasks
    
    def _create_instruction_tasks(self, n: int = 30) -> List[Dict]:
        """Create instruction-following tasks"""
        tasks = [
            {
                'prompt': 'List three colors.',
                'expected': ['red', 'blue', 'green', 'yellow', 'purple', 'orange'],
                'checker': lambda r: sum(1 for color in ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white'] if color in r.lower()) >= 3,
                'category': 'instruction'
            },
            {
                'prompt': 'Write a greeting.',
                'expected': ['hello', 'hi', 'greetings'],
                'checker': lambda r: any(greeting in r.lower() for greeting in ['hello', 'hi', 'greetings', 'welcome']),
                'category': 'instruction'
            },
            {
                'prompt': 'Count from 1 to 5.',
                'expected': ['1', '2', '3', '4', '5'],
                'checker': lambda r: all(str(i) in r for i in range(1, 6)),
                'category': 'instruction'
            }
        ]
        
        return tasks[:n]
    
    def _create_knowledge_tasks(self, n: int = 40) -> List[Dict]:
        """Create factual knowledge tasks"""
        tasks = [
            {
                'prompt': 'What is the capital of France?',
                'expected': 'Paris',
                'checker': lambda r: 'paris' in r.lower(),
                'category': 'knowledge'
            },
            {
                'prompt': 'How many days are in a week?',
                'expected': '7',
                'checker': lambda r: '7' in r or 'seven' in r.lower(),
                'category': 'knowledge'
            },
            {
                'prompt': 'What planet do we live on?',
                'expected': 'Earth',
                'checker': lambda r: 'earth' in r.lower(),
                'category': 'knowledge'
            }
        ]
        
        return tasks[:n]
    
    def _create_completion_tasks(self, n: int = 20) -> List[Dict]:
        """Create text completion tasks"""
        tasks = [
            {
                'prompt': 'The sun rises in the',
                'expected': 'east',
                'checker': lambda r: 'east' in r.lower(),
                'category': 'completion'
            },
            {
                'prompt': 'Water freezes at 0 degrees',
                'expected': 'celsius',
                'checker': lambda r: any(word in r.lower() for word in ['celsius', 'centigrade', 'c']),
                'category': 'completion'
            }
        ]
        
        return tasks[:n]
    
    def evaluate_model(self, model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> Dict[str, float]:
        """Evaluate model on proxy tasks"""
        
        model.eval()
        results = {}
        
        for task_type, task_list in self.tasks.items():
            correct = 0
            total = len(task_list)
            
            for task in tqdm(task_list, desc=f"Evaluating {task_type}", leave=False):
                # Generate response
                inputs = tokenizer(
                    task['prompt'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(task['prompt']):].strip()
                
                # Check if correct
                if task['checker'](response):
                    correct += 1
            
            results[task_type] = correct / total if total > 0 else 0.0
        
        return results


class ZeroCostEvaluator:
    """Main evaluation framework"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        
        # Initialize components
        self.statistical = StatisticalMetrics(self.config)
        self.proxy_tasks = ProxyTaskSuite(self.config)
        
        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(
        self,
        model: Union[GPT2LMHeadModel, str],
        tokenizer: Optional[GPT2TokenizerFast] = None,
        evaluation_texts: Optional[List[str]] = None,
        model_name: str = "unknown_model"
    ) -> Dict:
        """Comprehensive model evaluation"""
        
        # Load model if path provided
        if isinstance(model, str):
            model = GPT2LMHeadModel.from_pretrained(model)
            if tokenizer is None:
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
        
        # Move model to device
        model.to(self.config.device)
        model.eval()
        
        # Generate evaluation texts if not provided
        if evaluation_texts is None:
            evaluation_texts = self._generate_evaluation_texts(model, tokenizer)
        
        logger.info(f"Evaluating {model_name}...")
        results = {
            "model_name": model_name,
            "timestamp": time.time(),
            "model_size": sum(p.numel() for p in model.parameters()),
            "evaluation_config": self.config.__dict__
        }
        
        # 1. Statistical metrics
        logger.info("Computing statistical metrics...")
        results["perplexity"] = self.statistical.calculate_perplexity(model, evaluation_texts)
        results["diversity"] = self.statistical.calculate_diversity_metrics(evaluation_texts)
        results["quality"] = self.statistical.calculate_quality_proxies(evaluation_texts)
        
        # 2. Proxy tasks
        logger.info("Evaluating proxy tasks...")
        results["proxy_tasks"] = self.proxy_tasks.evaluate_model(model, tokenizer)
        
        # 3. Compute composite scores
        results["composite"] = self._compute_composite_scores(results)
        
        return results
    
    def _generate_evaluation_texts(self, model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> List[str]:
        """Generate texts for evaluation"""
        
        prompts = [
            "Write a short story about",
            "Explain how to",
            "The benefits of",
            "In my opinion",
            "Scientists have discovered",
            "The future of technology",
            "How to improve",
            "The importance of"
        ]
        
        texts = []
        
        for prompt in prompts:
            for _ in range(self.config.generation_samples // len(prompts)):
                inputs = tokenizer(prompt, return_tensors="pt").to(self.config.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                texts.append(text)
        
        return texts
    
    def _compute_composite_scores(self, results: Dict) -> Dict[str, float]:
        """Compute weighted composite scores"""
        
        composite = {}
        
        # Overall quality score
        quality_components = []
        
        if "quality" in results:
            quality_components.append(results["quality"].get("grammar_score", 0))
            quality_components.append(min(results["quality"].get("avg_readability", 0) / 100, 1.0))
            quality_components.append(results["quality"].get("coherence_proxy", 0))
        
        if quality_components:
            composite["quality_score"] = np.mean(quality_components)
        
        # Task performance
        if "proxy_tasks" in results:
            task_scores = list(results["proxy_tasks"].values())
            composite["task_performance"] = np.mean(task_scores) if task_scores else 0.0
        
        # Diversity score
        if "diversity" in results:
            div_scores = [
                results["diversity"].get("distinct_2", 0),
                results["diversity"].get("vocab_diversity", 0),
                min(results["diversity"].get("entropy", 0) / 10, 1.0)  # Normalize entropy
            ]
            composite["diversity_score"] = np.mean(div_scores)
        
        # Overall score (weighted combination)
        weights = {
            "quality_score": 0.4,
            "task_performance": 0.4,
            "diversity_score": 0.2
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in composite:
                overall_score += composite[metric] * weight
                total_weight += weight
        
        composite["overall_score"] = overall_score / total_weight if total_weight > 0 else 0.0
        
        return composite
    
    def save_results(self, results: Dict, output_path: Union[str, Path]):
        """Save evaluation results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Example usage of the zero-cost evaluation framework"""
    
    # Configuration
    config = EvaluationConfig(
        max_eval_samples=500,
        generation_samples=50,
        batch_size=8
    )
    
    # Initialize evaluator
    evaluator = ZeroCostEvaluator(config)
    
    # Example: Evaluate a pre-trained GPT-2 model
    print("Loading GPT-2 model for demonstration...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Run evaluation
    results = evaluator.evaluate_model(model, tokenizer, model_name="gpt2_baseline")
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Model: {results['model_name']}")
    print(f"Overall Score: {results['composite']['overall_score']:.3f}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Task Performance: {results['composite']['task_performance']:.3f}")
    print(f"Quality Score: {results['composite']['quality_score']:.3f}")
    print(f"Diversity Score: {results['composite']['diversity_score']:.3f}")
    
    # Save results
    evaluator.save_results(results, "evaluation_results.json")
    
    print("\nZero-cost evaluation complete!")


if __name__ == "__main__":
    main()