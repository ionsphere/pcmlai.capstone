"""
LLM Explanation Quality Evaluation

Provides tools for evaluating the quality of LLM-generated pricing explanations.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np


class ExplanationEvaluator:
    """
    Evaluates quality of LLM-generated pricing explanations.
    
    Metrics:
    1. Factual Accuracy: Does it correctly use provided data?
    2. Coherence: Is it well-structured and logical?
    3. Relevance: Does it focus on important pricing factors?
    4. Clarity: Is it easy to understand?
    5. Length: Is it appropriately concise?
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.evaluation_history = []
    
    def evaluate_factual_accuracy(
        self,
        explanation: str,
        ground_truth: Dict,
    ) -> Tuple[float, List[str]]:
        """
        Evaluate factual accuracy by checking if key facts are mentioned correctly.
        
        Args:
            explanation: Generated explanation text
            ground_truth: Dictionary with ground truth data
        
        Returns:
            Tuple of (score 0-1, list of issues)
        """
        issues = []
        checks_passed = 0
        total_checks = 0
        
        # Check 1: Condition score mentioned
        condition_score = ground_truth.get('condition_score')
        if condition_score is not None:
            total_checks += 1
            # Look for condition score or label
            if (str(int(condition_score)) in explanation or
                str(float(condition_score)) in explanation or
                self._condition_score_to_label(condition_score).lower() in explanation.lower()):
                checks_passed += 1
            else:
                issues.append("Condition score/label not mentioned")
        
        # Check 2: Price range mentioned
        price_range = ground_truth.get('predicted_price_range')
        if price_range is not None:
            total_checks += 1
            price_min, price_max = price_range
            # Look for prices (with some tolerance for rounding)
            if (self._find_price(explanation, price_min, tolerance=2) and
                self._find_price(explanation, price_max, tolerance=2)):
                checks_passed += 1
            else:
                issues.append("Price range not accurately mentioned")
        
        # Check 3: Market context mentioned
        pricing_context = ground_truth.get('pricing_context', {})
        if pricing_context:
            total_checks += 1
            mean_price = pricing_context.get('mean_price')
            n_similar = pricing_context.get('n_similar')
            
            # Check if market average mentioned
            if mean_price and self._find_price(explanation, mean_price, tolerance=5):
                checks_passed += 0.5
            
            # Check if similar items count mentioned
            if n_similar and str(n_similar) in explanation:
                checks_passed += 0.5
            
            if checks_passed < total_checks:
                issues.append("Market context insufficiently referenced")
        
        # Check 4: No contradictions with data
        total_checks += 1
        if not self._check_contradictions(explanation, ground_truth):
            checks_passed += 1
        else:
            issues.append("Contains contradictions with provided data")
        
        # Calculate score
        score = checks_passed / total_checks if total_checks > 0 else 0.0
        
        return score, issues
    
    def evaluate_coherence(self, explanation: str) -> Tuple[float, List[str]]:
        """
        Evaluate coherence and structure.
        
        Returns:
            Tuple of (score 0-1, list of issues)
        """
        issues = []
        score = 1.0
        
        # Check 1: Proper sentence structure
        sentences = self._split_sentences(explanation)
        if len(sentences) == 0:
            issues.append("No proper sentences found")
            return 0.0, issues
        
        # Check 2: Reasonable sentence length (not too short or long)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        if avg_sentence_length < 5:
            score -= 0.2
            issues.append("Sentences too short (fragmented)")
        elif avg_sentence_length > 40:
            score -= 0.2
            issues.append("Sentences too long (run-on)")
        
        # Check 3: Logical flow (has connecting words)
        connecting_words = ['however', 'therefore', 'thus', 'because', 'since', 'although', 
                           'additionally', 'moreover', 'furthermore', 'consequently']
        has_connections = any(word in explanation.lower() for word in connecting_words)
        if not has_connections and len(sentences) > 2:
            score -= 0.15
            issues.append("Lacks logical connectors between ideas")
        
        # Check 4: No repetitive phrases
        if self._has_repetition(explanation):
            score -= 0.15
            issues.append("Contains repetitive phrases")
        
        # Check 5: Proper capitalization and punctuation
        if not explanation[0].isupper():
            score -= 0.1
            issues.append("Does not start with capital letter")
        
        if not explanation.rstrip().endswith(('.', '!', '?')):
            score -= 0.1
            issues.append("Missing proper ending punctuation")
        
        return max(0.0, score), issues
    
    def evaluate_relevance(
        self,
        explanation: str,
        ground_truth: Dict,
    ) -> Tuple[float, List[str]]:
        """
        Evaluate relevance to pricing factors.
        
        Returns:
            Tuple of (score 0-1, list of issues)
        """
        issues = []
        relevance_keywords = {
            'condition': ['condition', 'wear', 'quality', 'state', 'pristine', 'damaged'],
            'market': ['market', 'similar', 'average', 'comparable', 'typical'],
            'price': ['price', 'cost', 'value', 'worth', 'priced'],
            'justification': ['because', 'due to', 'given', 'since', 'justified', 'appropriate', 'reasonable'],
        }
        
        scores = []
        for category, keywords in relevance_keywords.items():
            if any(kw in explanation.lower() for kw in keywords):
                scores.append(1.0)
            else:
                scores.append(0.0)
                issues.append(f"Missing {category} discussion")
        
        score = np.mean(scores)
        
        # Bonus: Addresses all three key areas
        if score >= 0.75:
            score = min(1.0, score + 0.1)
        
        return score, issues
    
    def evaluate_clarity(self, explanation: str) -> Tuple[float, List[str]]:
        """
        Evaluate clarity and readability.
        
        Returns:
            Tuple of (score 0-1, list of issues)
        """
        issues = []
        score = 1.0
        
        # Check 1: Reasonable length (50-200 words)
        word_count = len(explanation.split())
        if word_count < 30:
            score -= 0.3
            issues.append(f"Too short ({word_count} words, target: 50-150)")
        elif word_count > 200:
            score -= 0.2
            issues.append(f"Too long ({word_count} words, target: 50-150)")
        
        # Check 2: Not too technical (jargon level)
        technical_terms = ['quantile', 'percentile', 'deviation', 'coefficient', 'correlation']
        jargon_count = sum(1 for term in technical_terms if term in explanation.lower())
        if jargon_count > 2:
            score -= 0.2
            issues.append("Contains excessive technical jargon")
        
        # Check 3: Active voice (check for passive constructions)
        passive_indicators = [' is ', ' was ', ' are ', ' were ', ' been ', ' being ']
        passive_count = sum(explanation.lower().count(ind) for ind in passive_indicators)
        if passive_count > word_count * 0.15:  # More than 15% passive
            score -= 0.15
            issues.append("Overuse of passive voice")
        
        # Check 4: No excessive abbreviations
        abbr_pattern = r'\b[A-Z]{2,}\b'
        abbr_count = len(re.findall(abbr_pattern, explanation))
        if abbr_count > 3:
            score -= 0.15
            issues.append("Too many abbreviations")
        
        return max(0.0, score), issues
    
    def evaluate_comprehensive(
        self,
        explanation: str,
        ground_truth: Dict,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Comprehensive evaluation with all metrics.
        
        Args:
            explanation: Generated explanation text
            ground_truth: Ground truth data dictionary
            weights: Custom weights for each metric (default: equal)
        
        Returns:
            Dictionary with scores and detailed feedback
        """
        if weights is None:
            weights = {
                'factual_accuracy': 0.30,
                'coherence': 0.25,
                'relevance': 0.25,
                'clarity': 0.20,
            }
        
        # Run all evaluations
        accuracy_score, accuracy_issues = self.evaluate_factual_accuracy(explanation, ground_truth)
        coherence_score, coherence_issues = self.evaluate_coherence(explanation)
        relevance_score, relevance_issues = self.evaluate_relevance(explanation, ground_truth)
        clarity_score, clarity_issues = self.evaluate_clarity(explanation)
        
        # Calculate weighted overall score
        overall_score = (
            accuracy_score * weights['factual_accuracy'] +
            coherence_score * weights['coherence'] +
            relevance_score * weights['relevance'] +
            clarity_score * weights['clarity']
        )
        
        # Map to 5-point scale
        overall_score_5 = overall_score * 5.0
        
        # Compile results
        result = {
            'overall_score': overall_score,
            'overall_score_5': overall_score_5,
            'grade': self._score_to_grade(overall_score_5),
            'metrics': {
                'factual_accuracy': {
                    'score': accuracy_score,
                    'weight': weights['factual_accuracy'],
                    'issues': accuracy_issues,
                },
                'coherence': {
                    'score': coherence_score,
                    'weight': weights['coherence'],
                    'issues': coherence_issues,
                },
                'relevance': {
                    'score': relevance_score,
                    'weight': weights['relevance'],
                    'issues': relevance_issues,
                },
                'clarity': {
                    'score': clarity_score,
                    'weight': weights['clarity'],
                    'issues': clarity_issues,
                },
            },
            'explanation': explanation,
            'word_count': len(explanation.split()),
            'sentence_count': len(self._split_sentences(explanation)),
        }
        
        # Add to history
        self.evaluation_history.append(result)
        
        return result
    
    def _condition_score_to_label(self, score: float) -> str:
        """Convert condition score to label."""
        if score < 3:
            return "Poor"
        elif score < 5:
            return "Fair"
        elif score < 7:
            return "Good"
        elif score < 9:
            return "Very Good"
        else:
            return "Excellent"
    
    def _find_price(self, text: str, price: float, tolerance: float = 2.0) -> bool:
        """Check if price (within tolerance) is mentioned in text."""
        # Find all prices in format $XX.XX or $XX
        prices = re.findall(r'\$\s*(\d+(?:\.\d{2})?)', text)
        prices = [float(p) for p in prices]
        
        # Check if any price is within tolerance
        return any(abs(p - price) <= tolerance for p in prices)
    
    def _check_contradictions(self, explanation: str, ground_truth: Dict) -> bool:
        """Check for contradictions with ground truth."""
        # This is a simplified check - can be expanded
        contradictions = False
        
        # Check condition contradictions
        condition_score = ground_truth.get('condition_score')
        if condition_score is not None:
            label = self._condition_score_to_label(condition_score)
            # Check for opposite terms
            if condition_score > 7 and any(word in explanation.lower() for word in ['poor', 'bad', 'damaged', 'worn']):
                contradictions = True
            elif condition_score < 5 and any(word in explanation.lower() for word in ['excellent', 'pristine', 'perfect', 'like new']):
                contradictions = True
        
        return contradictions
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _has_repetition(self, text: str) -> bool:
        """Check for repetitive phrases (3+ words repeated)."""
        words = text.lower().split()
        # Check for 3-gram repetitions
        for i in range(len(words) - 5):
            trigram = ' '.join(words[i:i+3])
            rest = ' '.join(words[i+3:])
            if trigram in rest:
                return True
        return False
    
    def _score_to_grade(self, score: float) -> str:
        """Convert 5-point score to letter grade."""
        if score >= 4.5:
            return "A (Excellent)"
        elif score >= 4.0:
            return "B (Good)"
        elif score >= 3.0:
            return "C (Satisfactory)"
        elif score >= 2.0:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"
    
    def print_evaluation(self, result: Dict):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*80)
        print("EXPLANATION QUALITY EVALUATION")
        print("="*80)
        
        print(f"\nOverall Score: {result['overall_score_5']:.2f}/5.0 ({result['grade']})")
        print(f"Word Count: {result['word_count']}")
        print(f"Sentence Count: {result['sentence_count']}")
        
        print("\n--- Detailed Metrics ---")
        for metric_name, metric_data in result['metrics'].items():
            score = metric_data['score']
            weight = metric_data['weight']
            issues = metric_data['issues']
            
            print(f"\n{metric_name.replace('_', ' ').title()}:")
            print(f"  Score: {score:.2f}/1.0 (weight: {weight:.0%})")
            if issues:
                print(f"  Issues:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print(f"   No issues")
        
        print("\n--- Explanation ---")
        print(result['explanation'])
        print("\n" + "="*80)
    
    def save_evaluation_history(self, path: str):
        """Save evaluation history to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        
        print(f"Evaluation history saved to {path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics from evaluation history."""
        if not self.evaluation_history:
            return {"error": "No evaluations in history"}
        
        scores = [e['overall_score_5'] for e in self.evaluation_history]
        
        return {
            'n_evaluations': len(self.evaluation_history),
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'above_4_percent': np.mean([s >= 4.0 for s in scores]) * 100,
        }
