import json


def build_field_tree(fields):
    tree = {}
    for field in fields:
        node = tree
        for part in field.split("."):
            node = node.setdefault(part, {})
    return tree


def filter_data(data, field_tree):
    # 如果传进来的是 list，则自动转换
    if isinstance(field_tree, list):
        field_tree = build_field_tree(field_tree)

    if isinstance(data, dict):
        result = {}
        for key, subtree in field_tree.items():
            if key not in data:
                continue

            if not subtree:
                result[key] = data[key]
            else:
                value = filter_data(data[key], subtree)
                if value not in (None, {}, []):
                    result[key] = value
        return result

    elif isinstance(data, list):
        result = []
        for item in data:
            value = filter_data(item, field_tree)
            if value not in (None, {}, []):
                result.append(value)
        return result

    return data


if __name__ == '__main__':
    data = {
        "id": 1,
        "name": "Tom",
        "address": {
            "city": "Shanghai",
            "zip": "200000"
        },
        "orders": [
            {
                "id": 100,
                "price": 99,
                "items": [
                    {"name": "A", "count": 1},
                    {"name": "B", "count": 2}
                ]
            },
            {
                "id": 101,
                "price": 199,
                "items": [
                    {"name": "C", "count": 3}
                ]
            }
        ]
    }

    fields = [
        "name",
        "address.city",
        "orders.id",
        "orders.items.name"
    ]
    result = filter_data(data, fields)
    print(json.dumps(result, ensure_ascii=False, indent=4))
