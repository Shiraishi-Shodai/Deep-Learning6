from collections import defaultdict

text = "aaabaacab"
ids = list(text.encode("utf-8"))
# print(ids)

# ID列 : [97, 97, 97, 98, 97, 97, 99, 97, 98]
# 語彙サイズ : 256

# 隣接する整数値のペアのカウントを取る
def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        # print(f"pair: {pair}")
        counts[pair] += 1
    return counts

ids = [1, 2, 3, 1, 2]
counts = count_pairs(ids)
# print(counts)

# ペアの統合
def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    
    while i < len(ids):
        if i < len(ids) -1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    
    return merged_ids

ids = [1, 2, 3, 1, 2]
merged = merge(ids, (1, 2), 4)
# print(merged)

def train_bpe(text, vocab_size):
    # vocab_size : 最終的な語彙数
    
    # テキストを0 ~ 255に変換
    ids = list(text.encode("utf-8"))
    # print(f"text : {text} → encoding : {ids}")

    num_merges = vocab_size - 256
    merge_rules = {}

    for step in range(num_merges):
        counts = count_pairs(ids)

        if not counts:
            break
        
        # tupleが返る
        best_pair = max(counts, key=counts.get)

        new_id = 256 + step
        merge_rules[best_pair] = new_id
        
        # マージ実行
        ids = merge(ids, best_pair, new_id)

        return merge_rules

# 使用例
text = "Hello world! Thes is BPE training."

# BPEを学習
merge_rules = train_bpe(text, 260)
print(merge_rules)
