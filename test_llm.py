import os
from dotenv import load_dotenv

# Load env before importing provider
load_dotenv()

from llm_provider import get_llm

try:
    print(f"Model configured: {os.getenv('OPENAI_MODEL')}")
    print(f"API Key available: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    llm = get_llm()
    print("LLM instance created:", llm)
    
    result = llm.invoke("Hello, are you working? Respond with 'YES'.")
    print("LLM Response:", result.content)
except Exception as e:
    import traceback
    traceback.print_exc()
