import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# AgriChain AI - OpenAI Connection Test

BASE_DIR = Path(__file__).resolve().parent.parent

# Load API credentials from .env
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

print("\nAGRiCHAIN - OPENAI CONNECTION TEST")

if not api_key or api_key == "tu_api_key_aqui":
    raise ValueError(
        "OPENAI_API_KEY is missing. Check your .env file."
    )

print("API key detected successfully!")
print("Model:", model_name)

# Initialize OpenAI model
llm = ChatOpenAI(
    model=model_name,
    temperature=0,
    max_retries=2
)

print("\nConnecting to OpenAI...")

response = llm.invoke(
    "Reply with exactly this sentence: "
    "AgriChain AI is connected successfully."
)

print("\nOPENAI RESPONSE:")
print(response.content)

print("\nConnection test completed!")