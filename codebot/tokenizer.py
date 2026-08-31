from collections import defaultdict
import regex as re
from tqdm import tqdm
import time
import pickle

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
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
            
    return merged_ids

def pretokenize(text):
    # GPT-2で使用されている正規表現パターン
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(pattern, text)

# 1. 特殊トークンで分割
# 2. 事前トークン化
# 3. BPEの学習

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # ステップ1 : 特殊トークンで分割
    texts = input_text.split(end_token)

    # ステップ2 : 各テキスト片を事前トークン化
    ids_list = []
    for text in texts:
        for pretoken in pretokenize(text): # 事前トークン化(意味をもつ単語ごとに区切る)
            ids_list.append(list(pretoken.encode("utf-8"))) # 単語ごとにID列に変換
            # print(f"単語: {pretoken} → {list(pretoken.encode("utf-8"))}")
            
        num_merges = vocab_size - 256 - 1
        merge_rules = {}
    
    # print(f"ids_listの表示: {ids_list}, {len(ids_list)}")

    # マージしたい回数分ループを回す
    for step in tqdm(range(num_merges), desc="Training BPE"):
        counts = defaultdict(int)
        
        # 隣り合うペアとその回数を取得
        # 単語分割した単語の回数分count_pairsで隣り合うIDのペアを追加
        for ids in ids_list:
            counts = count_pairs(ids, counts)
        
        if not counts:
            break
        
        # 最も隣合う回数が多いペアを1つのID(整数)にまとめる
        best_pair = max(counts, key=counts.get)
        # 0 ~ 255で1byte分のID。最も隣り合うペアをマージするnew_idは256以降の数字
        new_id = 256 + step
        
        # マージルールを追加
        merge_rules[best_pair] = new_id
        
        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)
            
    return merge_rules
    
class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        # 特殊トークンのIDは256個の1byte分のID + マージIDから始まる
        self.end_token_id = 256 + len(merge_rules)
        
        # 0 ~ 255のIDとbyteの辞書を作る
        self.id_to_bytes = {i : bytes([i]) for i in range(256)}

        for (id1, id2), new_id in merge_rules.items():
            # b" He = b"H + b"e
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2] 
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")
        
        self.vocab_size = len(self.id_to_bytes)

    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)

        return ids
    
    def encode(self, input_text, show_progress=False):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []
    
        # show_progress=Trueならprogress表示
        texts = tqdm(texts, desc="Encoding") if show_progress else texts
        
        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                # 各事前トークンをBPEエンコード
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)
        
        return all_ids
        
    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")

        return text
    
    @staticmethod
    def load_from(filepath):
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)