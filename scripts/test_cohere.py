
import os
import cohere
from dotenv import load_dotenv

from indexing import load_vectorstores

from retrieval.bm25.retrieval import build_bm25_retriever
from retrieval.hybrid.retrieval import retrieve_hybrid_schemes
from retrieval.reranking.scheme_profile import build_scheme_profiles
from retrieval.reranking.cohere_rerank import rerank_schemes

load_dotenv()
# Initialize Cohere client
api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise ValueError("COHERE_API_KEY environment variable is missing.")

cohere_client = cohere.ClientV2(api_key=api_key)


# Load existing vector stores
discovery_db, _ = load_vectorstores()

# Build BM25 retriever
bm25_retriever = build_bm25_retriever(discovery_db)

query = "What government schemes are available for farmers?"


# Step 1: Hybrid retrieval
hybrid_result = retrieve_hybrid_schemes(
    query=query,
    discovery_db=discovery_db,
    bm25_retriever=bm25_retriever,
)

# Step 2: Build scheme profiles
profiles = build_scheme_profiles(
    discovery_db=discovery_db,
    fused_results=hybrid_result["retrieved_schemes"],
)

# Step 3: Cohere reranking
result = rerank_schemes(
    query=query,
    candidates=profiles,
    cohere_client=cohere_client,
    top_n=10,
)


# Display results
print("\nQuery:", result["query"])
print("Model:", result["model"])
print("Candidates:", result["candidates"])
print("Reranked results:", len(result["reranked_schemes"]))
print("Cohere latency:", round(result["latency"], 3), "seconds")

print("\nTop 10 Cohere Results:")

for scheme in result["reranked_schemes"]:
    print(
        f"Rank: {scheme['rank']} | "
        f"Scheme ID: {scheme['scheme_id']} | "
        f"Relevance: {scheme['relevance_score']:.4f} | "
        f"Original RRF Rank: {scheme['original_rrf_rank']}"
    )
