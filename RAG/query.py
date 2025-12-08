from sentence_transformers import SentenceTransformer
import chromadb
import torch

MODEL_PATH = r"models\qwen3-embedding"
DB_PATH = r"rag_db"
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16})
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("finance_rag")

def query_vector_db(query_text, top_k=5):
    """
    query_vector_db
    :param query_text: question
    :param top_k: top k results to return
    :return: answers
    """

    instruct_query = f"问题： {query_text}"

    query_embedding = model.encode(
        instruct_query,
        normalize_embeddings=True
    )
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    return results

def format_results(results):
    """
    format output
    """
    print("\n" + "="*80)
    print("results:")
    print("="*80)
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity = 1 - distance
        print(f"\n【结果 {i+1}】 相似度: {similarity:.4f}")
        print("-" * 80)
        print(f"内容: {doc[:500]}..." if len(doc) > 500 else f"内容: {doc}")
        print(f"元数据: {metadata}")
    
    print("\n" + "="*80)

# sample usage
if __name__ == "__main__":
    test_query = "什么是市盈率？如何用它评估股票？"
    results = query_vector_db(test_query, top_k=3)
    format_results(results)