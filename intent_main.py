import json

import numpy as np
import requests

BGE_URL = "http://127.0.0.1:11434/v1/embeddings"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer {Token}"
}


def get_embedding(texts):
    """
    texts: List[str]
    return: List[embedding]
    """
    payload = json.dumps({
        "model": "bge-m3",
        "input": texts,
        "encoding_format": "float"
    })

    response = requests.post(BGE_URL, headers=HEADERS, data=payload)
    data = response.json()["data"]

    return [item["embedding"] for item in data]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def intent_recognize(query: str, intent_list: list, threshold=0.75):
    """
    query: 用户输入文本
    intent_list: 意图列表
    threshold: 小于此阈值则返回 top3 推荐
    """

    # --- 1. 生成 query + 意图列表 的向量 ---
    texts = [query] + intent_list
    embeddings = get_embedding(texts)

    query_emb = embeddings[0]
    intent_embs = embeddings[1:]

    # --- 2. 计算所有意图与 query 的相似度 ---
    sims = []
    for i, emb in enumerate(intent_embs):
        sim = cosine_similarity(query_emb, emb)
        sims.append((intent_list[i], sim))

    # --- 3. 排序 ---
    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    best_intent, best_score = sims[0]

    # --- 4. 判断是否达到阈值 ---
    if best_score >= threshold:
        return {
            "type": "match",
            "intent": best_intent,
            "score": round(best_score, 4)
        }
    else:
        # 返回推荐 3 个意图
        top3 = sims[:3]
        return {
            "type": "recommend",
            "recommend": [
                {"intent": item[0], "score": round(item[1], 4)}
                for item in top3
            ]
        }


# ==========================
#       测试示例
# ==========================
if __name__ == "__main__":
    intent_list = [
        "查询天气",
        "播放音乐",
        "讲一个笑话",
        "打开日历",
        "帮我订餐",
        "订购机票",
        "订购高铁票"
    ]

    # query = "今天天气怎么样？"
    # query = "去北京要怎么坐车？"
    # query = "故事会里面会讲什么呢？"
    query = "订购高铁票？"

    result = intent_recognize(query, intent_list)
    print(json.dumps(result, ensure_ascii=False))
