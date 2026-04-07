import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {token}"}
res = requests.post(
    "https://router.huggingface.co/hf-inference/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
    headers=headers,
    json={"inputs": ["hello", "world"]}
)
print(res.status_code)
if res.status_code == 200:
    data = res.json()
    print(f"Success! Returned list of len {len(data)}")
else:
    print("Error:", res.json())
