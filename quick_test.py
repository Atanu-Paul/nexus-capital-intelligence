# quick_test_retrieval.py (temporary, delete after testing)
from rag.retriever import Retriever

retriever = Retriever()

results = retriever.retrieve("technology sector revenue growth", top_k=3)
for r in results:
    print(f"[{r['company']} - {r['doc_type']}] score: {r['score']:.3f}")
    print(r['chunk_text'][:150], "...\n")

print("--- Testing sector filter ---")
results2 = retriever.retrieve("company financial performance", top_k=3, sector_filter="banking")
for r in results2:
    print(f"[{r['company']} - {r['sector']}] score: {r['score']:.3f}")