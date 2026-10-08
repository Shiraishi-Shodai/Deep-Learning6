import os
import pickle
from multiprocessing import Pool
import shutil
from collections import defaultdict
import regex as re
from tqdm import tqdm
import numpy as np

def pretokenize(text):
    """事前トークン化
    """
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for m in re.finditer(pattern, text):
        yield m.group(0)

def count_pairs(ids, weight=1, counts=None):
    """同じ事前トークンがtext内に何個あったかを考慮してpairをカウント
    """
    if counts is None:
        counts = defaultdict(int)
    
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
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

def find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>"):
    byte_end_token = end_token.encode("utf-8")
    overlap = len(byte_end_token) - 1

    with open(file_path, "rb") as file:
        # ファイルサイズを取得
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks
        
        # チャンクの開始位置を計算(等間隔)
        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        # ファイル終端を追加
        chunk_boundaries.append(file_size)
        
        buffer_size = 4096 # 境界から先読みするバイト数(仮想メモリのサイズ)

        # 境界位置の調整(終了トークンを探す) 最初と最後のboundariesは無視
        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi]
            file.seek(chunk_position)

            while True:
                buffer = file.read(buffer_size)
                
                # ファイルに終端に達した場合
                if buffer == b"":
                    chunk_boundaries[bi] = file_size
                    break
                
                # 読み取ったチャンクで終了トークンを検索
                end_position = buffer.find(byte_end_token)
                if end_position != -1:
                    # 見つかった場合、その位置を新しい境界とする
                    chunk_boundaries[bi] = chunk_position + end_position
                    break
                    
                # 見つからなかった場合、次のバッファ位置に移動。overlapでend_tokenが跨らないように調整
                chunk_position += buffer_size - overlap
                # 調整した位置から次のbufferを読めるようにファイルポインタを移動
                file.seek(chunk_position)
    
    # 等間隔で作った開始位置をそれぞれ動かすため、開始位置の大小関係が崩れるたり、
    # たまたま開始位置がかぶる可能性がある。
    # これを整形する
    # 大小関係の崩壊例) [i, i + chunk_size] → [i + chunk_size + 10, i + chunk_size]
    # 開始位置の重複例) [i, i + chunk_size] → [i + chunk_size, i + chunk_size]
    return sorted(set(chunk_boundaries))

def process_single_chunk(file_path, start, end, end_token="<|endoftext|>"):
    pretoken_counts = defaultdict(int) # (key, value) = (token : string, token count : int)

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        # 特殊トークンで分割
        texts = chunk_text.split(end_token)

        # 事前トークン化
        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1
    
    return pretoken_counts

def pretoken_chunk(args):
    file_path, start, end, end_token = args
    pretoken_counts = defaultdict(int)

    # ファイルを開いてチャンクを読み込む
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        texts = chunk_text.split(end_token)

        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1
    
    return pretoken_counts


def train_bpe(file_path, vocab_size, end_token="<|endoftext|>", num_processes=8, num_chunks=8):
    assert vocab_size > 256 + 1,  "vocab_sizeは256+特殊トークン数より大きな値を設定してください。マージルールが作れません"
    
    # ステップ1: チャンクの準備
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
    total_chunk = len(chunk_boundaries) - 1
    
    chunk_info_list = []
    for i in range(total_chunk):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i+1]
        chunk_info_list.append((file_path, start, end, end_token))
    
    # ステップ2: 並列処理
    with Pool(processes=num_processes) as pool:
        all_results = list(tqdm(pool.imap(pretoken_chunk, chunk_info_list), total=len(chunk_info_list), desc="Pretokenizing"))

    # ステップ3: 事前トークン結果を統合
    pretoken_counts = defaultdict(int)
    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_counts[pretoken] += count

    
    # ステップ4: 事前トークンをbyte ID列に変換(ids_counts)
    # 例     : {
    #              [123, 213, 223, 12, 54] : 3,
    #              [215, 210, 134, 10, 84] : 5
    #          }
    ids_counts = {tuple(pretoken.encode("utf-8")) : count for pretoken, count in pretoken_counts.items()}

    # ステップ5: キャッシュにトークンを追加(pair_to_ids)
    # 例      : {
    #               (123, 213) : [123, 213, 223, 12, 54],
    #               (213, 223) : [123, 213, 223, 12, 54]
    #           }
    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    pair_to_ids = defaultdict(set) # キャッシュ
    
    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]): # キャッシュに登録
            pair_to_ids[pair].add(ids)

    # ステップ6: マージルールの生成
    #           キャッシュと事前トークンのカウントを更新
    for step in tqdm(range(num_merges), desc="Training BPE"):
        if not pair_counts: # ペアが存在しない場合
            break
            
        # 最頻出ペアを選択
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id
        
        # best_pairを含むids列をキャッシュから取得
        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair] # 使わないので削除
    
        # 影響のあるID列だけを更新
        # 更新対象
        #    ids_counts   : bytesとcountの辞書
        #    pair_to_ids  : キャッシュ
        for ids in affected_ids:
            ids_count = ids_counts[tuple(ids)]
            new_ids = merge(ids, best_pair, new_id)

            del ids_counts[tuple(ids)] # 古いIDを削除
            ids_counts[tuple(new_ids)] = ids_count # 新しいID列を追加
            
            # キャッシュから古いペア頻度を減少
            old_counts = count_pairs(ids)
            for pair, count in old_counts.items():
                pair_counts[pair] -= count * ids_count # 単語内の隣り合うIDが連続した回数 * ファイル内の単語の出現頻度
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                # 古いペアを持つトークンを一旦削除
                pair_to_ids[pair].discard(tuple(ids))

            # キャッシュに新しいペアを頻度を増加
            new_counts = count_pairs(new_ids)
            for pair, count in new_counts.items():
                pair_counts[pair] += count * ids_count # 単語内の隣り合うIDが連続した回数 * ファイル内の単語の出現頻度
                pair_to_ids[pair].add(tuple(new_ids))
    
    return merge_rules

class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)
        
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in self.merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    @staticmethod
    def load_from(filepath):
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)

    
    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))

        def get_merge_priority(pair):
            return self.merge_rules.get(pair, float('inf')) # 存在しないペアは最終優先度
        
        while len(ids) > 1:
            # ステップ1: 現在のペアを取得
            counts = count_pairs(ids)

            # ステップ2: 最優先ペアを特定(最初に作られたマージルールを優先)
            # 例:)
            # counts = {
            #     (1, 2) : 0,
            #     (2, 1) : 2,
            #     (1, 5) : 1,
            #     (100, 10) : 2,
            # }
            
            # merge_rules = {
            #     (1, 2) : 3,
            #     (2, 1) : 0,
            #     (1, 5) : 10
            # }
            
            # →(2, 1)のマージルールを取り出す
            best_pair = min(counts, key=get_merge_priority)

            # ステップ3: マージしきったらマージ後のidsを返す
            if best_pair not in self.merge_rules:
                break
            
            # ステップ4: マージ
            new_id = self.merge_rules[best_pair]
            ids = merge(ids, best_pair, new_id)
        
        return ids
    
    def encode(self, input_text, show_progress=False):
        # 例)
        # Hello<|endoftext|>World
        # ['Hello', '<|endoftext|>', 'World']
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []
        
        texts = tqdm(texts) if show_progress else texts
        
        # 例) 
        # ['Hello', '<|endoftext|>', 'World']
        # [72, 101, 108, 108, 111, 499, 87, 111, 114, 108, 100]
        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                # 各事前トークンをBPEエンコード
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)
        
        return all_ids
    
    def _encode_chunk(self, args):
        file_path, start, end, cache_dir, chunk_idx = args
        
        # チャンクをエンコード
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode("utf-8", errors="ignore")

            ids = self.encode(chunk_text)
        
        # ファイルから読み込んだテキストをトークンidとしてキャッシュファイルに保存
        # キャッシュファイルは、チャンクごとに作成
        cache_file = os.path.join(cache_dir, f"chunk{chunk_idx:05d}.npy")
        np.array(ids, dtype=np.unit16).tofile(cache_file)

        return cache_file, len(ids)