from indexing import load_vectorstores
from retrieval.dense.dense import retrieve_dense_schemes

discovery_db, _ = load_vectorstores()

query = "What government schemes are available for farmers?"

result = retrieve_dense_schemes(
    query=query,
    discovery_db=discovery_db,
)

print("Query:", result["query"])
print("Retrieved chunks:", result["retrieved_chunks"])
print("Unique schemes:", len(result["retrieved_schemes"]))
print("Latency:", round(result["latency"], 3), "seconds")

for scheme in result["retrieved_schemes"][:10]:
    print(scheme)
