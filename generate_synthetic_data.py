import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

def call_llm(prompt: str, output_file: str, num_sample_pairs: int = 1) -> list[str]:
    """Make API calls to gpt-5-nano via OpenRouter in batches of 10.
    
    Args:
        prompt: The prompt to send to the model
        num_samples: Number of samples to generate (default: 1)
    
    Returns:
        List of generated text samples
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    def make_request():
        """Make a single API request."""
        data = {
            "model": "openai/gpt-5-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        print(result)
        return result
    

    results = [make_request() for _ in range(num_sample_pairs)]
    
    # Write results to output_file as JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

prompt = """
Generate exactly a pair of sentences, one with an angry tone and one with a neutral tone. You should generate exactly two sentences total. Examples of contrastive pairs:

ANGER: “I’m furious that this happened.”

NEUTRAL: “I heard that this happened.”

ANGER: “This makes me so angry.”

NEUTRAL: “This reminds me of something.”

ANGER: “I’m absolutely livid at you.”

NEUTRAL: “I’m absolutely certain about that.”

Do not just copy or rephrase the examples above. Generate original sentences.
"""

if __name__ == "__main__":
    results = call_llm(prompt, "data/test.json", num_sample_pairs=25)

