import json
from pathlib import Path

# AgriChain AI - Knowledge Base Inspection

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "raw" / "knowledge_base.json"

with open(KB_PATH, "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print("\nAGRiCHAIN - KNOWLEDGE BASE INSPECTION")

print("\nROOT DATA TYPE:")
print(type(knowledge_base).__name__)

if isinstance(knowledge_base, list):
    print("\nTOTAL DOCUMENTS:", len(knowledge_base))

    if knowledge_base:
        print("\nFIRST DOCUMENT:")
        print(json.dumps(knowledge_base[0], indent=2))

elif isinstance(knowledge_base, dict):
    print("\nROOT KEYS:")
    print(list(knowledge_base.keys()))

    for key, value in knowledge_base.items():
        print(f"\n{key}: {type(value).__name__}")

        if isinstance(value, list):
            print("Number of items:", len(value))

            if value:
                print("First item:")
                print(json.dumps(value[0], indent=2)[:3000])

print("\nInspection completed successfully!")