import json
from data_ingestion.loaders.scheme_loader import load_schemes

schemes = load_schemes()

print("Total schemes:", len(schemes))
print("\nFirst scheme:")
print(json.dumps(schemes[0], indent=2, ensure_ascii=False))

print("\nFields across first scheme:")
print(list(schemes[0].keys()))