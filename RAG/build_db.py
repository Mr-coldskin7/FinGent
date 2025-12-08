import json
import os
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import torch
import gc

DATA_PATH = r"data\deepseek-fin\datasets-rpXruFMUY6-T-alpaca-2025-06-16.json"
MODEL_PATH = r"models\qwen3-embedding"
DB_PATH = r"rag_db"
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16})

def load_data():
    documents = []
    metadatas = []
    ids = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        datalist = json.load(f)
        for idx, data in enumerate(datalist):
            instruction = data.get('instruction', '')
            output = data.get('output', '')
            doc = f"问题：{instruction}\n答案：{output}"
            documents.append(doc)
            metadatas.append({"source": "deepseek-fin", "type": "qa"})
            ids.append(str(idx))
    return documents, metadatas, ids

def build_vector_db():
    """构建向量数据库（分块处理，边生成边存储）"""
    documents, metadatas, ids = load_data()
    print(f"共加载 {len(documents)} 条问答")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("finance_rag")
    except:
        pass
    collection = client.create_collection(
        name="finance_rag",
        metadata={"hnsw:space": "cosine"}
    )
    chunk_size = 20
    print(f"\n开始分块处理，每块 {chunk_size} 条，边生成边存储...")
    
    for i in tqdm(range(0, len(documents), chunk_size), desc="处理进度"):
        chunk_docs = documents[i:i + chunk_size]
        chunk_metadatas = metadatas[i:i + chunk_size]
        chunk_ids = ids[i:i + chunk_size]
        chunk_embeddings = model.encode(
            chunk_docs,
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        collection.add(
            embeddings=chunk_embeddings.tolist(),
            documents=chunk_docs,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        
        del chunk_embeddings
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n✅ 向量库构建完成！共 {len(documents)} 条记录")
    print(f"最终显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB 已分配")

if __name__ == "__main__":
    # 如果GPU仍然失败，取消下方注释强制使用CPU（慢但稳定）
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    build_vector_db()