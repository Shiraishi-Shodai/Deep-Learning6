import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from collections import defaultdict
import regex as re
from tqdm import tqdm

def pretokenize(text):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(pattern, text)

def count_pair(ids, weights=1, counts=None):
    if counts is None:
        counts = defaultdict(int)
    
    for pair in zip(ids, ids[1:]):
        counts[pair] += weights
    
    return counts

def merge(ids, best_pair, new_id):
    i = 0
    merged_ids = []
    
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    
    return merged_ids

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # end tokenで分割
    texts = input_text.split(end_token)

    # 各トークンのカウントを計算
    pretoken_counts = defaultdict(int)
    for text in tqdm(texts, desc="Pretokenize"):
        for pretoken in pretokenize(text):
            pretoken_counts[pretoken] += 1
    
    # 事前トークンをID化
    ids_counts = {tuple(p.encode("utf-8")) : c for p, c in pretoken_counts.items()}
    
    # マージルール作成準備
    merge_rules = {}
    num_merges = vocab_size - 256 - 1
    pair_to_ids = defaultdict(set) # キャッシュ
    
    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pair(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]): # キャッシュに登録
            pair_to_ids[pair].add(ids)

    # for pair, ids in pair_to_ids.items():
    #     print(f"pair: {pair}, ids: {ids}")
    
    for step in tqdm(range(num_merges), desc="Training BPE"):
        if not pair_counts:
            break
        
        # 最頻出ペアを算出
        best_pair = max(pair_counts, key=lambda pair : (pair_counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id
        
        # best_pairを含むid列をキャッシュから取得(キャッシュからbest_pairを削除)
        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair]
        
        # 影響のあるID列だけを更新
        for ids in affected_ids:
            ids_count = ids_counts[tuple(ids)]
            new_ids = merge(ids, best_pair, new_id)
            # 古いペアを削除
            del ids_counts[tuple(ids)]
            # 新しいペアとカントを追加
            ids_counts[tuple(new_ids)] = ids_count

            # 古いペア頻度を減少
            old_counts = count_pair(ids)
            for pair, count in old_counts.items():
                pair_counts[pair] -= count * ids_count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_ids[pair].discard(tuple(ids))

            # 新しいペア頻度を増加
            new_counts = count_pair(new_ids)
            for pair, count in new_counts.items():
                pair_counts[pair] += count * ids_count
                pair_to_ids[pair].add(tuple(new_ids))
        
    return merge_rules

vocab_size = 1000
file_path = "codebot/tiny_codes.txt"
text = open(file_path).read()
merge_rules = train_bpe(input_text=text, vocab_size=vocab_size)
print(merge_rules)