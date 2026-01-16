# -*- coding: utf-8 -*-
"""
用户问题聚类（固定 20 类）
Embedding: Ollama bge-m3
Clustering: KMeans
"""

import json
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# 1. Ollama Embedding 配置
# =========================
OLLAMA_BASE_URL = "http://localhost:11434/v1/embeddings"
MODEL_NAME = "bge-m3:latest"


def get_embeddings(texts):
    payload = {
        "model": MODEL_NAME,
        "input": texts
    }
    resp = requests.post(OLLAMA_BASE_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()["data"]
    return np.array([d["embedding"] for d in data], dtype="float32")


# =========================
# 2. 读取问题
# =========================
def extract_texts(json_path):
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

# L2 归一化（非常重要）
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# =========================
# 4. KMeans 固定 20 个聚类
# =========================
print("Clustering with KMeans (20 clusters)...")

NUM_CLUSTERS = 20
kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init="auto"
)

labels = kmeans.fit_predict(embeddings)

# =========================
# 5. 按类别分组
# =========================
clustered = defaultdict(list)
for q, label in zip(questions, labels):
    clustered[label].append(q)


# =========================
# 6. 提取聚类关键词
# =========================
def extract_keywords(texts, top_k=10):
    vectorizer = TfidfVectorizer(
        max_features=50,
        ngram_range=(1, 2),
        stop_words=None
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

for label in sorted(clustered.keys()):
    qs = clustered[label]
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
df.to_csv("question_clusters_20.csv", index=False, encoding="utf-8-sig")

print("\n结果已保存：question_clusters_20.csv")
