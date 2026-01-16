# -*- coding: utf-8 -*-
"""
用户问题聚类（Ollama bge-m3 + OpenAI Embeddings 规范）
"""

import requests
import numpy as np
import pandas as pd
import hdbscan
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# 1. Ollama Embedding 配置
# =========================
OLLAMA_BASE_URL = "http://localhost:11434/v1/embeddings"
MODEL_NAME = "bge-m3:latest"


def get_embeddings(texts):
    """
    使用 OpenAI 兼容规范调用 Ollama bge-m3
    """
    payload = {
        "model": MODEL_NAME,
        "input": texts
    }

    resp = requests.post(OLLAMA_BASE_URL, json=payload, timeout=60)
    resp.raise_for_status()

    data = resp.json()["data"]
    embeddings = [item["embedding"] for item in data]

    return np.array(embeddings, dtype="float32")


# =========================
# 2. 示例用户问题
# =========================
def extract_texts(json_path):
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        turn["text"]
        for item in data
        for turn in item.get("turns", [])
        if "text" in turn
    ]
json_path = "./chinese_conversations_dataset/q_final.json"
questions = extract_texts(json_path)

# =========================
# 3. 获取 Embedding
# =========================
print("Getting embeddings from Ollama (bge-m3)...")
embeddings = get_embeddings(questions)

# 可选：L2 归一化（强烈推荐）
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

# =========================
# 4. HDBSCAN 聚类
# =========================
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    metric="euclidean"
)

labels = clusterer.fit_predict(embeddings)

# =========================
# 5. 按类别分组
# =========================
clustered = defaultdict(list)
for q, label in zip(questions, labels):
    clustered[label].append(q)


# =========================
# 6. 提取聚类关键词
# =========================
def extract_keywords(texts, top_k=5):
    vectorizer = TfidfVectorizer(
        max_features=30,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    scores = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    ranked = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]


# =========================
# 7. 输出结果
# =========================
print("\n========== 聚类结果 ==========")

for label, qs in clustered.items():
    if label == -1:
        print("\n【未归类问题】")
    else:
        keywords = extract_keywords(qs)
        print(f"\n【类别 {label} | 关键词: {', '.join(keywords)}】")

    for q in qs:
        print(f" - {q}")

# =========================
# 8. 保存结果
# =========================
df = pd.DataFrame({
    "question": questions,
    "cluster": labels
})
df.to_csv("question_clusters.csv", index=False, encoding="utf-8-sig")

print("\n结果已保存：question_clusters.csv")
