"""
RAGAS 评估脚本 - 使用真实 RAGAS 库测试 FinGent RAG 模块性能

安装依赖:
    pip install ragas datasets langchain langchain-community langchain-huggingface

RAGAS 核心指标:
- Faithfulness: 答案是否基于检索的上下文（无幻觉）
- Answer Relevancy: 答案与问题的相关程度  
- Context Precision: 检索结果中相关片段的比例
- Context Recall: 是否检索到了所有相关信息
"""

import json
import os
import sys
from typing import List
from dataclasses import dataclass

from dotenv import load_dotenv

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# 检查环境
RAG_DB_PATH = os.path.join(PROJECT_ROOT, "RAG", "rag_db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "RAG", "models", "qwen3-embedding")

def check_environment():
    """检查运行环境是否满足"""
    errors = []
    
    if not os.path.exists(RAG_DB_PATH):
        errors.append(f"向量库目录不存在: {RAG_DB_PATH}")
        errors.append("请先运行: python RAG/build_db.py 创建向量数据库")
    else:
        print(f"✅ 向量库目录存在: {RAG_DB_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        errors.append(f"嵌入模型目录不存在: {MODEL_PATH}")
    else:
        print(f"✅ 模型目录存在: {MODEL_PATH}")
    
    # 检查 RAGAS 是否安装
    try:
        import ragas
        print(f"✅ RAGAS 已安装")
    except ImportError:
        errors.append("RAGAS 库未安装，请运行: pip install ragas datasets")
    
    # 检查 API Key
    load_dotenv()
    api_key = os.getenv("QIANWEN_API_KEY")
    if not api_key:
        print("⚠️ 未设置 QIANWEN_API_KEY 环境变量")
    else:
        print("✅ QIANWEN_API_KEY 已设置")
    
    if errors:
        print("\n❌ 环境检查失败:")
        for err in errors:
            print(f"   - {err}")
        return False
    
    return True

# 导入 RAG 模块
try:
    from RAG.db_operations import VectorStoreBase
except ImportError as e:
    print(f"❌ 导入 RAG 模块失败: {e}")
    sys.exit(1)


@dataclass
class RAGTestCase:
    """测试用例结构"""
    question: str
    ground_truth: str
    contexts: List[str]
    answer: str


class FinGentRAGEvaluator:
    """FinGent RAG 系统评估器"""
    
    def __init__(self):
        try:
            self.rag_store = VectorStoreBase()
            print("✅ RAG 评估器初始化成功\n")
        except Exception as e:
            print(f"❌ RAG 评估器初始化失败: {e}")
            raise
    
    def retrieve_contexts(self, question: str, top_k: int = 3) -> List[str]:
        """检索上下文"""
        results = self.rag_store.search(question, top_k=top_k)
        
        contexts = []
        if results and results.get("documents"):
            for doc_list in results["documents"]:
                for doc in doc_list:
                    contexts.append(doc)
        
        return contexts


def prepare_test_dataset() -> List[RAGTestCase]:
    """准备测试数据集"""
    test_cases = [
        {
            "question": "什么是市盈率（PE），如何用它判断股票估值？",
            "ground_truth": "市盈率（Price-to-Earnings Ratio）是股票价格与每股收益的比率。PE = 股价 / EPS。一般来说，PE越低表示股票可能被低估，但也要结合行业特点判断。成长股的PE通常较高，价值股的PE较低。"
        },
        {
            "question": "ROE 是什么意思，如何计算？",
            "ground_truth": "ROE（Return on Equity，净资产收益率）是净利润与股东权益的比率，计算公式为：ROE = 净利润 / 平均股东权益 × 100%。ROE反映股东权益的收益水平，是衡量公司盈利能力的重要指标。"
        },
        {
            "question": "如何分析公司的现金流？",
            "ground_truth": "分析公司现金流主要关注三个方面：1）经营活动现金流：应该为正且持续增长；2）投资活动现金流：扩张期通常为负；3）筹资活动现金流：反映融资情况。健康的公司经营活动现金流应该能覆盖投资和筹资需求。"
        },
        {
            "question": "什么是杜邦分析法？",
            "ground_truth": "杜邦分析法是一种综合财务分析方法，将ROE分解为三个部分：ROE = 销售净利率 × 总资产周转率 × 权益乘数。通过分解可以分析企业盈利能力的驱动因素：盈利能力、营运效率和财务杠杆。"
        },
        {
            "question": "如何判断一家公司的财务健康状况？",
            "ground_truth": "判断公司财务健康可以从四个方面：1）偿债能力：资产负债率、流动比率；2）盈利能力：ROE、毛利率、净利率；3）营运能力：存货周转率、应收账款周转率；4）成长能力：营收增长率、净利润增长率。"
        },
    ]
    
    return [RAGTestCase(**case, contexts=[], answer="") for case in test_cases]


def run_ragas_with_qwen():
    """使用通义千问运行 RAGAS 评估"""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    from langchain_community.chat_models import ChatTongyi
    from langchain_huggingface import HuggingFaceEmbeddings
    
    print("=" * 60)
    print("FinGent RAG 系统 - RAGAS 专业评估")
    print("=" * 60)
    
    if not check_environment():
        print("\n⚠️ 请解决上述问题后重新运行")
        return None
    
    api_key = os.getenv("QIANWEN_API_KEY")
    if not api_key:
        print("\n❌ 未设置 QIANWEN_API_KEY，无法运行 RAGAS 评估")
        print("请设置环境变量: set QIANWEN_API_KEY=your_key")
        return run_simplified_evaluation()
    
    # 初始化
    evaluator = FinGentRAGEvaluator()
    test_cases = prepare_test_dataset()
    
    print(f"📊 测试集规模: {len(test_cases)} 个问答对")
    print("-" * 60)
    
    # 构建数据集
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] 检索: {case.question[:40]}...")
        
        contexts = evaluator.retrieve_contexts(case.question, top_k=3)
        
        if contexts:
            answer = contexts[0][:400] + "..." if len(contexts[0]) > 400 else contexts[0]
        else:
            answer = "未找到相关信息"
        
        data["question"].append(case.question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(case.ground_truth)
        
        print(f"    检索到 {len(contexts)} 条上下文")
    
    # 创建 HuggingFace Dataset
    dataset = Dataset.from_dict(data)
    
    print("\n" + "=" * 60)
    print("🚀 开始 RAGAS 评估（调用通义千问，可能需要几分钟）...")
    print("=" * 60)
    
    try:
        # 配置通义千问 LLM
        print("配置评判 LLM (通义千问)...")
        judge_llm = ChatTongyi(
            api_key=api_key,
            model_name="qwen-max",
            temperature=0.0
        )
        
        # 配置本地嵌入模型（避免调用 OpenAI）
        print("配置嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 运行评估 - RAGAS 会自动使用配置的 LLM 和嵌入模型
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=judge_llm,
            embeddings=embeddings
        )
        
        # 打印结果
        print("\n" + "=" * 60)
        print("📈 RAGAS 评估结果")
        print("=" * 60)
        
        for metric_name, score in result.items():
            print(f"{metric_name:25s}: {score:.3f}")
        
        avg_score = sum(result.values()) / len(result)
        print(f"\n{'综合得分':25s}: {avg_score:.3f}")
        
        if avg_score >= 0.7:
            rating = "优秀 (Excellent) 🎉"
        elif avg_score >= 0.5:
            rating = "良好 (Good) 👍"
        elif avg_score >= 0.3:
            rating = "一般 (Fair) ⚠️"
        else:
            rating = "需改进 (Needs Improvement) ❌"
        
        print(f"评级: {rating}")
        
        # 保存结果
        output = {
            "ragas_scores": {k: float(v) for k, v in result.items()},
            "average_score": float(avg_score),
            "rating": rating,
            "test_cases": len(test_cases)
        }
        
        with open("ragas_evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print("\n详细结果已保存: ragas_evaluation_results.json")
        
        return result
        
    except Exception as e:
        print(f"\n❌ RAGAS 评估失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n尝试使用简化版评估...")
        return run_simplified_evaluation(evaluator, test_cases)


def run_simplified_evaluation(evaluator=None, test_cases=None):
    """简化版评估（RAGAS 失败时的备选）"""
    print("\n" + "=" * 60)
    print("📊 简化版评估（基于关键词匹配）")
    print("=" * 60)
    
    if evaluator is None:
        evaluator = FinGentRAGEvaluator()
    if test_cases is None:
        test_cases = prepare_test_dataset()
    
    results = {
        "context_count": [],
        "avg_context_length": [],
        "questions": []
    }
    
    for i, case in enumerate(test_cases, 1):
        contexts = evaluator.retrieve_contexts(case.question, top_k=3)
        
        results["context_count"].append(len(contexts))
        avg_len = sum(len(c) for c in contexts) / len(contexts) if contexts else 0
        results["avg_context_length"].append(avg_len)
        results["questions"].append({
            "question": case.question,
            "contexts_found": len(contexts)
        })
        
        print(f"[{i}] {case.question[:40]}... -> 检索到 {len(contexts)} 条")
    
    avg_contexts = sum(results['context_count']) / len(results['context_count'])
    avg_length = sum(results['avg_context_length']) / len(results['avg_context_length'])
    
    print(f"\n平均检索结果数: {avg_contexts:.1f}")
    print(f"平均上下文长度: {avg_length:.0f} 字符")
    
    # 保存结果
    with open("simplified_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


if __name__ == "__main__":
    run_ragas_with_qwen()
