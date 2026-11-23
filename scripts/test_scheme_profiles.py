
from indexing import load_vectorstores

from retrieval.bm25.retrieval import build_bm25_retriever
from retrieval.hybrid.retrieval import retrieve_hybrid_schemes
from retrieval.reranking.scheme_profile import build_scheme_profiles


discovery_db, _ = load_vectorstores()

bm25_retriever = build_bm25_retriever(discovery_db)

query = "What government schemes are available for farmers?"

hybrid_result = retrieve_hybrid_schemes(
    query=query,
    discovery_db=discovery_db,
    bm25_retriever=bm25_retriever,
)

profiles = build_scheme_profiles(
    discovery_db=discovery_db,
    fused_results=hybrid_result["retrieved_schemes"],
)

print("Hybrid candidates:", len(hybrid_result["retrieved_schemes"]))
print("Generated profiles:", len(profiles))

print("\nFirst 3 Scheme Profiles:")

for profile in profiles[:3]:
    print("\n" + "=" * 60)
    print("Scheme ID:", profile["scheme_id"])
    print("RRF Rank:", profile["rank"])
    print("Profile characters:", len(profile["text"]))
    print("Profile preview:\n", profile["text"][:500])
