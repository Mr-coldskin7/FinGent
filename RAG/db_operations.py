import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
import torch

# =============================================================================
MODEL_PATH = r"models\qwen3-embedding"
DB_PATH = r"rag_db"
COLLECTION_NAME = "finance_rag"
# =============================================================================


class VectorStoreBase:
    """
    base class for vector store and operations
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        db_path: str = DB_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer(
            MODEL_PATH, trust_remote_code=True, model_kwargs={"dtype": torch.float16}
        )
        self.collection = self.client.get_collection(name=collection_name)

    def search(self, query_text, top_k=5):
        """
        query_vector_db
        :param query_text: question
        :param top_k: top k results to return
        :return: answers
        """
        instruct_query = f"问题： {query_text}"

        query_embedding = self.model.encode(instruct_query, normalize_embeddings=True)

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def add(self, texts, metadatas=None, ids=None, batch_size=1):
        """
        add info to vector db
        :param texts: input texts
        :param metadata: optional
        :param ids: optional
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if metadatas is None:
            metadatas = [{"source": "deepseek-fin", "type": "qa"}] * len(texts)

        if ids is None:
            ids = [f"doc_{hashlib.md5(t.encode()).hexdigest()[:16]}" for t in texts]
        elif isinstance(ids, str):
            ids = [ids]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        del embeddings
        torch.cuda.empty_cache()
        print(f"Added {len(texts)} texts to vector database.")

    def format_results(results):
        """
        format output
        """
        print("\n" + "=" * 80)
        print("results:")
        print("=" * 80)

        for i, (doc, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            similarity = 1 - distance
            print(f"\n【结果 {i+1}】 相似度: {similarity:.4f}")
            print("-" * 80)
            print(f"内容: {doc[:500]}..." if len(doc) > 500 else f"内容: {doc}")
            print(f"元数据: {metadata}")

        print("\n" + "=" * 80)


# sample usage
if __name__ == "__main__":
    test_query = ["这是测试数据", "这不是测试数据"]
    vector_store = VectorStoreBase()
    vector_store.add(test_query)
    results = vector_store.search(test_query[1], top_k=1)
    VectorStoreBase.format_results(results)
