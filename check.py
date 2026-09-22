import torch
import torch.nn as nn
from collections import defaultdict
import re

text = "hello世界😄"
# print(list(text))
# print(ord("h"))
# print(ord("😄"))

# print(chr(104))
# print(chr(128516))

# ids = [ord(char) for char in list(text)]
# # print(ids)

# class CharTokenizer:
#     def encode(self, text):
#         return [ord(char) for char in list(text)]
    
#     def decode(self, ids):
#         return [chr(id) for id in ids]

# tokenizer = CharTokenizer()
# encoded = tokenizer.encode(text)
# decoded = tokenizer.decode(encoded)

# print(encoded)
# print(decoded)

"""02_byte_tokenizer.py
"""
class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))
    
    def decode(self, ids):
        return bytes(ids).decode("utf-8")

# encoded = 'A'.encode("utf-8")
# print(encoded)
# print(list(encoded))
# encoded = "あ".encode("utf-8")
# print(encoded)
# print(list(encoded))

# print(type(bytes(list(encoded)).decode("utf-8")))
# print(type(bytes(list(encoded))))

# tokenizer = ByteTokenizer()
# text = "hello世界😆"
# ids = tokenizer.encode(text)
# decoded = tokenizer.decode(ids)

# print(ids)
# print(decoded)

"""03_bpe_train.py
"""
def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

# ids = [1, 2, 3, 1, 2]
# counts = count_pairs(ids)
# print(counts)

def merge(ids, pair, new_id):
    merge_ids = []
    i = 0
    
    while i < len(ids):
        if i < len(ids) -1 and (ids[i], ids[i+1]) == pair:
            merge_ids.append(new_id)
            i += 2
        else:
            merge_ids.append(ids[i])
            i += 1
    return merge_ids


ids = [1, 2, 3, 1, 2]
merged = merge(ids, (1, 2), 4)
# print(merged) # [4, 4, 2]

def train_bpe(text, vocab_size):
    ids = list(text.encode("utf-8"))

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

# a = defaultdict(int)
# a[(1, 2)] = 10
# a[(0, 1)] = 1
# print(max(a, key=a.get))

# 使用例
text = "Hello world! Thes is BPE training."

# BPEを学習
# merge_rules = train_bpe(text, 260)
# print(merge_rules)

"""04_bpe_tokenizer.py
"""
class BPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules
        
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        
        for(id1, id2), new_id in self.merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
            self.vocab_size = len(self.id_to_bytes)
    
    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        
        return ids

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

# サンプルマージルール
merge_rules = {(105, 115) : 256, (256, 32) : 257,
               (105, 110) : 258, (72, 101) : 259}

# トークナイザ作成
# tokenizer = BPETokenizer(merge_rules)
# text = "Hello世界😆"
# ids = tokenizer.encode(text)
# decoded = tokenizer.decode(ids)

# print(ids)
# print(decoded)
# print(tokenizer.id_to_bytes)

"""05_special_tokenizer.py
"""
# text =  "a/b"
# pattern = "(" + re.escape("/") + ")"
# result = re.split(pattern, text)
# print(result)

# a = [1, 2]
# b = [2, 3]
# a.extend(b)
# print(a)