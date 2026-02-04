#!/usr/bin/env python3
"""
Script to test and compare available models from Tata Communications API.
Tests models based on:
- Response time
- Response correctness
- Token usage
- Error rates
"""

import os
import json
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL = "https://models.cloudservices.tatacommunications.com/v1"
API_KEY = os.getenv("GROK_API_KEY", "")

# Known models from the codebase
KNOWN_MODELS = [
    "meta/Llama-3.1-8B-Instruct",
    "meta/llama-3.1-70b-instruct",
    "openai/gpt-oss-120b",
    "openai/gpt-4o-mini",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-Embedding-8B",
]

# Common model patterns to try
COMMON_MODEL_PATTERNS = [
    "meta/Llama-3.1-8B-Instruct",
    "meta/Llama-3.1-70B-Instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3-8b-instruct",
    "meta/llama-3-70b-instruct",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-3.5-turbo",
    "Qwen/Qwen3-Embedding-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-7B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct",
    "google/gemma-7b-it",
    "google/gemma-2b-it",
]


class ModelTester:
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = API_BASE_URL
        self.results: List[Dict[str, Any]] = []
        self.available_models: List[str] = []
        
    def list_models(self) -> List[str]:
        """Try to list available models from /v1/models endpoint"""
        print("\n" + "="*80)
        print("🔍 Attempting to list available models...")
        print("="*80)
        
        if not self.api_key:
            print("⚠️  No API key found. Will try without authentication.")
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Successfully retrieved models list!")
                print(json.dumps(data, indent=2))
                
                # Extract model IDs
                if isinstance(data, dict) and "data" in data:
                    models = [m.get("id", "") for m in data.get("data", [])]
                    return [m for m in models if m]
                elif isinstance(data, list):
                    return [m.get("id", "") if isinstance(m, dict) else str(m) for m in data]
            else:
                print(f"❌ Failed to list models: {response.status_code}")
                print(f"Response: {response.text[:500]}")
        except Exception as e:
            print(f"❌ Error listing models: {e}")
        
        return []
    
    def test_model(
        self, 
        model: str, 
        test_prompt: str = "What is 2+2? Answer with just the number.",
        expected_answer: str = "4"
    ) -> Dict[str, Any]:
        """Test a single model and return performance metrics"""
        result = {
            "model": model,
            "status": "unknown",
            "response_time": None,
            "response_code": None,
            "error": None,
            "tokens_used": None,
            "response_text": None,
            "correct": False,
            "timestamp": datetime.now().isoformat()
        }
        
        if not self.api_key:
            result["status"] = "skipped"
            result["error"] = "No API key"
            return result
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": test_prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0
            )
            
            elapsed_time = time.time() - start_time
            result["response_time"] = round(elapsed_time, 3)
            result["response_code"] = response.status_code
            
            if response.status_code == 200:
                data = response.json()
                result["status"] = "success"
                
                    # Extract response text
                if "choices" in data and len(data["choices"]) > 0:
                    result["response_text"] = data["choices"][0].get("message", {}).get("content", "").strip()
                    
                    # Check correctness (more flexible - extract numbers)
                    response_lower = result["response_text"].lower()
                    # Check if expected answer appears, or extract first number
                    if expected_answer.lower() in response_lower:
                        result["correct"] = True
                    else:
                        # Try to extract number from response
                        import re
                        numbers = re.findall(r'\d+', result["response_text"])
                        if numbers and numbers[0] == expected_answer:
                            result["correct"] = True
                
                # Extract token usage
                if "usage" in data:
                    result["tokens_used"] = {
                        "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                        "completion_tokens": data["usage"].get("completion_tokens", 0),
                        "total_tokens": data["usage"].get("total_tokens", 0)
                    }
            else:
                result["status"] = "error"
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                
        except requests.Timeout:
            result["status"] = "timeout"
            result["error"] = "Request timeout"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def run_comprehensive_test(self):
        """Run comprehensive tests on all models"""
        print("\n" + "="*80)
        print("🚀 Starting Comprehensive Model Testing")
        print("="*80)
        
        # First, try to get list of available models
        available_models = self.list_models()
        
        # Combine with known models
        models_to_test = list(set(KNOWN_MODELS + COMMON_MODEL_PATTERNS))
        if available_models:
            models_to_test.extend(available_models)
            models_to_test = list(set(models_to_test))
        
        print(f"\n📋 Testing {len(models_to_test)} models...")
        print(f"Models: {', '.join(models_to_test[:10])}{'...' if len(models_to_test) > 10 else ''}\n")
        
        # Test each model
        for i, model in enumerate(models_to_test, 1):
            print(f"[{i}/{len(models_to_test)}] Testing {model}...", end=" ", flush=True)
            result = self.test_model(model)
            self.results.append(result)
            
            if result["status"] == "success":
                tokens = result.get('tokens_used', {})
                total_tokens = tokens.get('total_tokens', 'N/A') if isinstance(tokens, dict) else 'N/A'
                print(f"✅ Success ({result['response_time']}s, {total_tokens} tokens)")
            elif result["status"] == "error":
                print(f"❌ Error: {result.get('error', 'Unknown')[:50]}")
            elif result["status"] == "timeout":
                print(f"⏱️  Timeout")
            else:
                print(f"⚠️  {result['status']}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        self.print_results()
    
    def print_results(self):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print("📊 TEST RESULTS SUMMARY")
        print("="*80)
        
        # Filter successful results
        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] != "success"]
        
        print(f"\n✅ Successful: {len(successful)}/{len(self.results)}")
        print(f"❌ Failed: {len(failed)}/{len(self.results)}")
        
        if not successful:
            print("\n⚠️  No successful tests. Check API key and network connectivity.")
            return
        
        # Sort by response time
        successful.sort(key=lambda x: x.get("response_time", float("inf")))
        
        print("\n" + "-"*80)
        print("🏆 TOP PERFORMING MODELS (by response time)")
        print("-"*80)
        print(f"{'Model':<50} {'Time (s)':<12} {'Tokens':<15} {'Correct':<10}")
        print("-"*80)
        
        for result in successful[:10]:
            tokens = result.get("tokens_used", {})
            total_tokens = tokens.get("total_tokens", "N/A") if isinstance(tokens, dict) else "N/A"
            correct = "✅" if result.get("correct") else "❌"
            print(f"{result['model']:<50} {result['response_time']:<12.3f} {str(total_tokens):<15} {correct:<10}")
        
        # Sort by tokens (efficiency)
        print("\n" + "-"*80)
        print("💰 MOST EFFICIENT MODELS (by token usage)")
        print("-"*80)
        print(f"{'Model':<50} {'Total Tokens':<15} {'Time (s)':<12} {'Correct':<10}")
        print("-"*80)
        
        efficient = [r for r in successful if r.get("tokens_used") and isinstance(r["tokens_used"], dict)]
        efficient.sort(key=lambda x: x.get("tokens_used", {}).get("total_tokens", float("inf")))
        
        for result in efficient[:10]:
            tokens = result.get("tokens_used", {})
            total_tokens = tokens.get("total_tokens", "N/A")
            correct = "✅" if result.get("correct") else "❌"
            print(f"{result['model']:<50} {str(total_tokens):<15} {result['response_time']:<12.3f} {correct:<10}")
        
        # Detailed results
        print("\n" + "-"*80)
        print("📋 DETAILED RESULTS")
        print("-"*80)
        
        for result in successful:
            print(f"\n🤖 Model: {result['model']}")
            print(f"   Status: {result['status']}")
            print(f"   Response Time: {result['response_time']}s")
            print(f"   Response Code: {result['response_code']}")
            if result.get("tokens_used"):
                tokens = result["tokens_used"]
                print(f"   Tokens - Prompt: {tokens.get('prompt_tokens', 'N/A')}, "
                      f"Completion: {tokens.get('completion_tokens', 'N/A')}, "
                      f"Total: {tokens.get('total_tokens', 'N/A')}")
            print(f"   Correct: {'✅ Yes' if result.get('correct') else '❌ No'}")
            if result.get("response_text"):
                print(f"   Response: {result['response_text'][:100]}...")
        
        if failed:
            print("\n" + "-"*80)
            print("❌ FAILED MODELS")
            print("-"*80)
            for result in failed:
                print(f"\n❌ {result['model']}")
                print(f"   Status: {result['status']}")
                print(f"   Error: {result.get('error', 'Unknown')}")
        
        # Save to JSON
        output_file = "model_test_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")


def main():
    if not API_KEY:
        print("⚠️  WARNING: GROK_API_KEY not found in environment.")
        print("   Set it with: export GROK_API_KEY='your-key'")
        print("   Or add it to your .env file")
        print("\n   Continuing anyway to test API discovery...\n")
    
    tester = ModelTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()
