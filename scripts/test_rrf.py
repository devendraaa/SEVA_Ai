
from indexing import load_vectorstores

from retrieval.dense.dense import retrieve_dense_schemes
from retrieval.bm25.retrieval import (build_bm25_retriever, retrieve_bm25_schemes )
from retrieval.fusion.rrf import reciprocal_rank_fusion


discovery_db, _ = load_vectorstores()

bm25_retriever = build_bm25_retriever(discovery_db)

query = "What government schemes are available for farmers?"

dense_result = retrieve_dense_schemes(query, discovery_db)

bm25_result = retrieve_bm25_schemes(
    query,
    bm25_retriever,
)

fused_results = reciprocal_rank_fusion(
    dense_result["retrieved_schemes"],
    bm25_result["retrieved_schemes"],
)

print("Dense schemes:", len(dense_result["retrieved_schemes"]))
print("BM25 schemes:", len(bm25_result["retrieved_schemes"]))
print("Fused schemes:", len(fused_results))

print("\nTop 10 RRF results:")

for result in fused_results[:10]:
    print(result)
