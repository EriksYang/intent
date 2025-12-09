import json

import numpy as np
import requests

BGE_URL = "http://127.0.0.1:11434/v1/embeddings"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer {Token}"
}


def get_embedding(texts):
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


def intent_recognize(query: str, intent_list: list, threshold=0.75, top=3):
    """
    返回所有达到阈值的意图，不只返回一个
    """

    # 1. 对 query 和所有意图进行向量化
    intents = [f'{item["intent"]},{item["description"]}' for item in intent_list]
    texts = [query] + intents
    embeddings = get_embedding(texts)

    query_emb = embeddings[0]
    intent_embs = embeddings[1:]

    # 2. 计算相似度
    sims = []
    for i, emb in enumerate(intent_embs):
        sim = cosine_similarity(query_emb, emb)
        sims.append((intent_list[i], sim))

    # 3. 根据相似度排序
    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    # 4. 找出所有达到阈值的
    hits = [item for item in sims if item[1] >= threshold]

    if hits:
        # 全部返回
        return {
            "type": "match",
            "intents": [
                {
                    "intent": item[0],
                    "score": round(item[1], 4)
                }
                for item in hits[:top]
            ]
        }
    else:
        # 没有达到阈值 → 返回 top3
        return {
            "type": "recommend",
            "recommend": [
                {
                    "intent": item[0],
                    "score": round(item[1], 4)}
                for item in sims[:top]
            ]
        }


# ==============
#   测试案例
# ==============
if __name__ == "__main__":
    intent_list = [
        {
            "intent": "查询天气",
            "description": "用于获取当前或未来的天气情况。"
        },
        {
            "intent": "播放音乐",
            "description": "用于播放指定歌曲、歌单或音乐类型。"
        },
        {
            "intent": "讲一个笑话",
            "description": "用于让系统随机讲述一个轻松有趣的笑话。"
        },
        {
            "intent": "打开日历",
            "description": "用于打开日历查看日期、日程安排等信息。"
        },
        {
            "intent": "帮我订餐",
            "description": "用于根据用户需求下单外卖或餐食预定。"
        },
        {
            "intent": "订购机票",
            "description": "用于查询并预订航班机票。"
        },
        {
            "intent": "订购高铁票",
            "description": "用于查询并购买高铁车票。"
        }
    ]
    # query = "查询一下今天北京未来15天的天气"
    # query = "去北京要怎么坐车？"
    # query = "故事会里面会讲什么呢？"
    # query = "我需要买高铁票去北京，然后买飞机票回上海"
    query = "我需要订购高铁票"

    result = intent_recognize(query, intent_list, threshold=0.75, top=3)
    print(json.dumps(result, ensure_ascii=False, indent=2))
