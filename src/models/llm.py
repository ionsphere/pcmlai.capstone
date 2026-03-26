import json
import requests
from typing import Dict, List, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import time


class LLMBackend(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        request_timeout: int = 180,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.request_timeout = request_timeout
    
    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            result = response.json()
            models = result.get("models", [])
            return [
                model.get("name")
                for model in models
                if isinstance(model, dict) and model.get("name")
            ]
        except requests.exceptions.RequestException:
            return []
    
    def resolve_model_name(self, requested_model: str) -> str:
        available_models = self.list_models()
        if not available_models:
            return requested_model
        
        if requested_model in available_models:
            return requested_model
        
        normalized = requested_model.strip().lower()
        exact_base_matches = [
            model for model in available_models
            if model.split(":", 1)[0].lower() == normalized
        ]
        if exact_base_matches:
            return exact_base_matches[0]
        
        prefix_matches = [
            model for model in available_models
            if model.lower().startswith(normalized)
        ]
        if prefix_matches:
            return prefix_matches[0]
        
        return requested_model
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        payload = {
            "model": self.resolve_model_name(self.model_name),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=self.request_timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            available_models = self.list_models()
            available_text = f" Installed models: {', '.join(available_models)}." if available_models else ""
            if isinstance(e, requests.exceptions.ReadTimeout):
                raise RuntimeError(
                    f"Ollama API timed out after {self.request_timeout}s while generating with "
                    f"'{self.resolve_model_name(self.model_name)}'. Try a smaller/faster model or "
                    f"lower max_tokens.{available_text}"
                )
            raise RuntimeError(f"Ollama API error: {e}.{available_text}")
    
    def is_available(self) -> bool:
        return len(self.list_models()) > 0


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-3.5-turbo",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.api_url = f"{base_url}/chat/completions"
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API error: {e}")
    
    def is_available(self) -> bool:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            response = requests.get(f"{self.base_url}/models", headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False


class TransformersBackend(LLMBackend):
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            kwargs = {"device_map": self.device}
            if self.load_in_8bit:
                kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                kwargs["load_in_4bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **kwargs
            )
            
            print(f"Loaded {self.model_name} on {self.device}")
        except ImportError:
            raise RuntimeError("transformers library not installed. Install with: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        except Exception as e:
            raise RuntimeError(f"Generation error: {e}")
    
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None


class LLMExplainer:
    def __init__(
        self,
        backend: LLMBackend,
        prompt_template: Optional[str] = None,
    ):
        self.backend = backend
        self.prompt_template = prompt_template or self._default_prompt_template()
        
        if not self.backend.is_available():
            print("Warning: LLM backend may not be available. Check configuration.")
    
    def _default_prompt_template(self) -> str:
       return """You are an expert in second-hand clothing pricing. Given the following information about a clothing item, provide a concise explanation (3-5 sentences) for why the predicted price range is appropriate.

Item Information:
- Clothing Type: {clothing_type}
- Condition Score: {condition_score}/10 ({condition_label})
- Predicted Price Range: ${price_min:.2f} - ${price_max:.2f}

Market Context:
- Similar Items Found: {n_similar}
- Average Price of Similar Items: ${mean_price:.2f}
- Price Range of Similar Items: ${similar_min:.2f} - ${similar_max:.2f}
- Market Confidence: {confidence:.0%}

Top Similar Items:
{similar_items}

Provide a clear, factual explanation focusing on:
1. How the condition affects the price
2. How the item compares to similar items in the market
3. Why this price range is reasonable

Explanation:"""
    
    def generate_explanation(
        self,
        clothing_type: str,
        condition_score: float,
        predicted_price_range: tuple,
        pricing_context: Dict,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Dict[str, Union[str, float]]:
        condition_labels = {
            (1, 3): "Poor - Significant wear",
            (3, 5): "Fair - Moderate wear",
            (5, 7): "Good - Light wear",
            (7, 9): "Very Good - Minimal wear",
            (9, 11): "Excellent - Like new"
        }
        condition_label = "Unknown"
        for (low, high), label in condition_labels.items():
            if low <= condition_score < high:
                condition_label = label
                break
        
        similar_items_text = ""
        if "similar_items" in pricing_context:
            for i, item in enumerate(pricing_context["similar_items"][:3], 1):
                similar_items_text += f"  {i}. {item.get('category', 'Item')} - ${item.get('price', 0):.2f} (Similarity: {item.get('similarity', 0):.0%})\n"
        else:
            similar_items_text = "  (No similar items available)\n"
        
        prompt = self.prompt_template.format(
            clothing_type=clothing_type,
            condition_score=condition_score,
            condition_label=condition_label,
            price_min=predicted_price_range[0],
            price_max=predicted_price_range[1],
            n_similar=pricing_context.get("n_similar", 0),
            mean_price=pricing_context.get("mean_price", 0),
            similar_min=pricing_context.get("min_price", 0),
            similar_max=pricing_context.get("max_price", 0),
            confidence=pricing_context.get("confidence", 0),
            similar_items=similar_items_text.strip(),
        )
        
        start_time = time.time()
        try:
            explanation = self.backend.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            generation_time = time.time() - start_time
            
            return {
                "explanation": explanation,
                "generation_time": generation_time,
                "model": getattr(self.backend, "model_name", "unknown"),
                "prompt": prompt,
                "success": True,
            }
        except Exception as e:
            return {
                "explanation": f"Error generating explanation: {str(e)}",
                "generation_time": time.time() - start_time,
                "model": getattr(self.backend, "model_name", "unknown"),
                "prompt": prompt,
                "success": False,
                "error": str(e),
            }
    
    def set_prompt_template(self, template: str):
        self.prompt_template = template
    
    def save_prompt_template(self, path: str):
        Path(path).write_text(self.prompt_template)
        print(f"Prompt template saved to {path}")
    
    def load_prompt_template(self, path: str):
        self.prompt_template = Path(path).read_text()
        print(f"Prompt template loaded from {path}")
