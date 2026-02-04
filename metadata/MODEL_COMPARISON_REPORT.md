# Model Comparison Report
## Tata Communications API: https://models.cloudservices.tatacommunications.com/v1/chat/completions

**Date:** February 3, 2026  
**API Endpoint:** `https://models.cloudservices.tatacommunications.com/v1/chat/completions`

---

## 📋 Available Models

Based on the `/v1/models` endpoint, the following **9 models** are available:

### Chat Completion Models (7):
1. **meta/Llama-3.1-8B-Instruct** ✅
2. **meta/Llama-4-Scout-17B-16E-Instruct** ✅
3. **meta/Llama-3.3-70B-Instruct** ✅
4. **openai/gpt-oss-20b** ✅
5. **openai/gpt-oss-120b** ✅
6. **Qwen/Qwen2.5-Coder-14B-Instruct** ✅
7. **test-meta/Llama-4-Scout-17B-16E-Instruct** ✅ (test variant)

### Embedding Models (2):
- **Qwen/Qwen3-Embedding-8B** (embedding only, not chat)
- **avsolatorio/GIST-Embedding-v0** (embedding only, not chat)

---

## 🏆 Performance Comparison

### Fastest Response Times (Top 5)

| Rank | Model | Response Time | Total Tokens | Status |
|------|-------|---------------|--------------|--------|
| 🥇 1 | **meta/Llama-3.1-8B-Instruct** | **0.179s** | 49 | ✅ Fastest |
| 🥈 2 | **Qwen/Qwen2.5-Coder-14B-Instruct** | **0.230s** | 43 | ✅ Very Fast |
| 🥉 3 | **meta/Llama-3.3-70B-Instruct** | **0.297s** | 50 | ✅ Fast |
| 4 | **openai/gpt-oss-20b** | 0.345s | 125 | ✅ Moderate |
| 5 | **test-meta/Llama-4-Scout-17B-16E-Instruct** | 0.381s | 25 | ✅ Moderate |

### Most Token-Efficient Models (Top 5)

| Rank | Model | Total Tokens | Response Time | Efficiency |
|------|-------|--------------|---------------|------------|
| 🥇 1 | **test-meta/Llama-4-Scout-17B-16E-Instruct** | **25** | 0.381s | ⭐⭐⭐⭐⭐ |
| 🥈 2 | **meta/Llama-4-Scout-17B-16E-Instruct** | **25** | 0.466s | ⭐⭐⭐⭐⭐ |
| 🥉 3 | **Qwen/Qwen2.5-Coder-14B-Instruct** | **43** | 0.230s | ⭐⭐⭐⭐ |
| 4 | **meta/Llama-3.1-8B-Instruct** | 49 | 0.179s | ⭐⭐⭐⭐ |
| 5 | **meta/Llama-3.3-70B-Instruct** | 50 | 0.297s | ⭐⭐⭐⭐ |

### Detailed Model Analysis

#### 1. **meta/Llama-3.1-8B-Instruct** ⭐⭐⭐⭐⭐
- **Response Time:** 0.179s (Fastest)
- **Token Usage:** 49 tokens (Prompt: 47, Completion: 2)
- **Response Code:** 200 ✅
- **Strengths:**
  - Fastest response time
  - Low token consumption
  - Currently used as PRIMARY_CHAT_MODEL in your codebase
- **Best For:** General purpose, fast responses, cost-effective
- **Recommendation:** ✅ **EXCELLENT CHOICE** - Best balance of speed and efficiency

#### 2. **Qwen/Qwen2.5-Coder-14B-Instruct** ⭐⭐⭐⭐⭐
- **Response Time:** 0.230s (Very Fast)
- **Token Usage:** 43 tokens (Prompt: 41, Completion: 2)
- **Response Code:** 200 ✅
- **Strengths:**
  - Second fastest
  - Most token-efficient chat model
  - Specialized for coding tasks
- **Best For:** Code generation, technical queries, programming assistance
- **Recommendation:** ✅ **EXCELLENT CHOICE** - Best for coding-related tasks

#### 3. **meta/Llama-3.3-70B-Instruct** ⭐⭐⭐⭐
- **Response Time:** 0.297s (Fast)
- **Token Usage:** 50 tokens (Prompt: 47, Completion: 3)
- **Response Code:** 200 ✅
- **Strengths:**
  - Large model (70B parameters) - likely better reasoning
  - Still fast response time
  - Good token efficiency
- **Best For:** Complex reasoning, detailed analysis, high-quality responses
- **Recommendation:** ✅ **GOOD CHOICE** - Best for quality when speed is acceptable

#### 4. **meta/Llama-4-Scout-17B-16E-Instruct** ⭐⭐⭐⭐
- **Response Time:** 0.466s (Moderate)
- **Token Usage:** 25 tokens (Prompt: 22, Completion: 3) - **Most Efficient!**
- **Response Code:** 200 ✅
- **Strengths:**
  - Most token-efficient (lowest token usage)
  - Latest Llama 4 model
- **Best For:** Cost optimization, simple queries
- **Recommendation:** ✅ **GOOD CHOICE** - Best for minimizing token costs

#### 5. **openai/gpt-oss-20b** ⭐⭐⭐
- **Response Time:** 0.345s (Moderate)
- **Token Usage:** 125 tokens (Prompt: 81, Completion: 44)
- **Response Code:** 200 ✅
- **Strengths:**
  - Moderate speed
- **Weaknesses:**
  - Higher token consumption (2.5x more than Llama-3.1-8B)
- **Best For:** General purpose (but less efficient than alternatives)
- **Recommendation:** ⚠️ **MODERATE** - Higher cost due to token usage

#### 6. **openai/gpt-oss-120b** ⭐⭐⭐
- **Response Time:** 0.433s (Moderate)
- **Token Usage:** 114 tokens (Prompt: 77, Completion: 37)
- **Response Code:** 200 ✅
- **Strengths:**
  - Very large model (120B parameters) - potentially best reasoning
- **Weaknesses:**
  - Slower response time
  - Higher token consumption
- **Best For:** Complex reasoning tasks where quality > speed/cost
- **Recommendation:** ⚠️ **MODERATE** - Use only when maximum quality is needed

#### 7. **test-meta/Llama-4-Scout-17B-16E-Instruct** ⭐⭐⭐
- **Response Time:** 0.381s (Moderate)
- **Token Usage:** 25 tokens (Prompt: 22, Completion: 3)
- **Response Code:** 200 ✅
- **Note:** This is a test variant - may not be stable for production
- **Recommendation:** ⚠️ **NOT RECOMMENDED** - Use production version instead

---

## 📊 Summary Statistics

### Response Time Distribution:
- **Fastest:** 0.179s (meta/Llama-3.1-8B-Instruct)
- **Slowest:** 0.466s (meta/Llama-4-Scout-17B-16E-Instruct)
- **Average:** ~0.33s
- **All models respond in < 0.5s** ✅

### Token Usage Distribution:
- **Most Efficient:** 25 tokens (Llama-4-Scout models)
- **Least Efficient:** 125 tokens (gpt-oss-20b)
- **Average:** ~61 tokens

### Success Rate:
- **7/7 available chat models:** 100% success rate ✅
- **All models return HTTP 200** ✅
- **No timeouts or connection errors** ✅

---

## 🎯 Recommendations

### For General Purpose Use:
**🥇 Recommended: `meta/Llama-3.1-8B-Instruct`**
- Fastest response time (0.179s)
- Good token efficiency (49 tokens)
- Currently your primary model
- Best balance of speed, cost, and quality

### For Coding/Technical Tasks:
**🥇 Recommended: `Qwen/Qwen2.5-Coder-14B-Instruct`**
- Very fast (0.230s)
- Most token-efficient (43 tokens)
- Specialized for coding tasks
- Excellent for technical queries

### For High-Quality Reasoning:
**🥇 Recommended: `meta/Llama-3.3-70B-Instruct`**
- Large model (70B) for better reasoning
- Still fast (0.297s)
- Good token efficiency (50 tokens)
- Best for complex analysis

### For Cost Optimization:
**🥇 Recommended: `meta/Llama-4-Scout-17B-16E-Instruct`**
- Lowest token usage (25 tokens)
- Moderate speed (0.466s)
- Best for minimizing API costs

### Fallback Order (Current in your codebase):
Your current fallback order is:
1. `meta/Llama-3.1-8B-Instruct` (PRIMARY) ✅
2. `meta/llama-3.1-70b-instruct` ❌ (Not available - returns error)
3. `openai/gpt-oss-120b` ✅
4. `openai/gpt-4o-mini` ❌ (Not available)

**Suggested Updated Fallback Order:**
1. `meta/Llama-3.1-8B-Instruct` (PRIMARY) ✅
2. `Qwen/Qwen2.5-Coder-14B-Instruct` ✅ (Fast, efficient)
3. `meta/Llama-3.3-70B-Instruct` ✅ (Quality fallback)
4. `openai/gpt-oss-120b` ✅ (Last resort for complex tasks)

---

## ⚠️ Models NOT Available

The following models tested are **NOT available** on this API:
- `openai/gpt-3.5-turbo`
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `meta/llama-3.1-70b-instruct` (lowercase variant)
- `meta/Llama-3.1-70B-Instruct` (uppercase variant)
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`
- `mistralai/Mistral-7B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct`
- `google/gemma-2b-it`
- `google/gemma-7b-it`

---

## 📝 Notes

1. **Case Sensitivity:** Model names are case-sensitive. Use exact names from `/v1/models` endpoint.
2. **Embedding Models:** `Qwen/Qwen3-Embedding-8B` and `avsolatorio/GIST-Embedding-v0` are for embeddings only, not chat completions.
3. **Test Model:** `test-meta/Llama-4-Scout-17B-16E-Instruct` is a test variant - prefer production version.
4. **All models tested successfully** - No connection or authentication issues.
5. **Response Quality:** All models responded correctly to test queries (though format may vary).

---

## 🔧 Implementation Suggestions

Update your `ai_service.py` fallback models:

```python
PRIMARY_CHAT_MODEL = "meta/Llama-3.1-8B-Instruct"  # ✅ Keep as primary

FALLBACK_CHAT_MODELS = [
    "Qwen/Qwen2.5-Coder-14B-Instruct",  # ✅ Fast, efficient, coding-focused
    "meta/Llama-3.3-70B-Instruct",      # ✅ Quality fallback
    "openai/gpt-oss-120b",               # ✅ Complex reasoning fallback
]
```

---

**Report Generated:** February 3, 2026  
**Test Script:** `test_models.py`  
**Results File:** `model_test_results.json`
