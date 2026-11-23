from indexing import load_vectorstores
from retrieval.bm25.retrieval import (
    build_bm25_retriever,
    retrieve_bm25_schemes
    )

discovery_db, _ = load_vectorstores()

bm25_retriever = build_bm25_retriever(discovery_db)

query = "What government schemes are available for farmers?"

result = retrieve_bm25_schemes(
    query=query,
    bm25_retriever=bm25_retriever,
)

print("Query:", result["query"])
print("Retrieved chunks:", result["retrieved_chunks"])
print("Unique schemes:", len(result["retrieved_schemes"]))
print("Latency:", round(result["latency"], 3), "seconds")

for scheme in result["retrieved_schemes"][:10]:
    print(scheme)
