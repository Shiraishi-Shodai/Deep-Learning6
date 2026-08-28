from collections import defaultdict
import itertools
import re

def count_pairs(ids, counts=None):
    
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    
    return counts

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    
    return merged_ids
        

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    texts = input_text.split(end_token)
    ids_list = [list(text.encode("utf-8")) for text in texts]
    
    # 基本語彙 + 特殊トークンを除いた値がマージ回数
    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    
    for step in range(num_merges):

        counts = defaultdict(int)
        # 隣り合うidのペアをカウント
        for ids in ids_list:
            counts = count_pairs(ids, counts)
        
        # if len(counts) == 0 :
        #     break
        
        best_pair = max(counts, key=counts.get)
        new_id = 256 + step

        merge_rules[best_pair] = new_id

        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)
        
    return merge_rules
        

merge_rules = train_bpe("Hello World!<|endoftext|>This is BPE training", 260)
# print(merge_rules)

class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)
        self.id_to_bytes = {id : bytes([id]) for id in range(256)}

        for (id1, id2), new_id in self.merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")
        self.vocab_size = len(self.id_to_bytes)
    
    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merge_rules.items():
            ids = merge(ids, pair, new_id)
        return ids
    
    def encode(self, text):
        pattern = "(" + re.escape(self.end_token) + ")"
        texts = re.split(pattern, text)
        all_ids = []
        
        for i in range(len(texts)):
            all_ids.append(self._encode_text(texts[i]))

        return all_ids 

    def decode(self, ids):

        byte_list = [self.id_to_bytes[id] for id in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        
        return text
    
print(merge_rules)
tokenizer = BPETokenizer(merge_rules, end_token="<|endoftext|>")
encoded = tokenizer.encode("Hello World!<|endoftext|>This is BPE training")
decoded = "".join([tokenizer.decode(ids) for ids in encoded])
print(decoded)