from collections import defaultdict

text = "aaabaacab"
ids = list(text.encode("utf-8"))
print(ids)

# ID列 : [97, 97, 97, 98, 97, 97, 99, 97, 98]
# 語彙サイズ : 256

# 隣接する整数値のペアのカウントを取る
def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        print(f"pair: {pair}")
        counts[pair] += 1
    return counts

ids = [1, 2, 3, 1, 2]
counts = count_pairs(ids)
print(counts)

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
print(merged)

def train_bpe(text, vocab_size):
    # vocab_size : 最終的な語彙数
    
    # テキストを0 ~ 255に変換
    ids = list(text.encode("utf-8"))

    num_merges = vocab_size - 256
    merge_rules = {}

    