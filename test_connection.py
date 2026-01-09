import os
import openai
from dotenv import load_dotenv

load_dotenv()

MODELS_TO_TEST = [
    "openai/gpt-4o-mini",
    "openai/gpt-3.5-turbo",
    "meta/llama-3.1-70b-instruct",
    "openai/gpt-oss-120b",
]

def test_models():
    client = openai.OpenAI(
        base_url=os.getenv("GROK_BASE_URL"),
        api_key=os.getenv("GROK_API_KEY"),
        timeout=15
    )
    
    print("Testing models...")
    working = []
    failed = []
    
    for model in MODELS_TO_TEST:
        try:
            print(f"Testing {model}...", end=" ")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
                timeout=15
            )
            if resp and resp.choices:
                print("✅ WORKING")
                working.append(model)
            else:
                print("❌ FAILED (no response)")
                failed.append(model)
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:100]}")
            failed.append(model)
    
    print("\n=== RESULTS ===")
    print(f"Working models ({len(working)}): {working}")
    print(f"Failed models ({len(failed)}): {failed}")
    
    if working:
        print(f"\n✅ Use this in .env:\nCHAT_MODEL={working[0]}")

if __name__ == "__main__":
    test_models()