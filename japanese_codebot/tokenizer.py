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

    
    # 事前トークンをbyte ID列に変換
    ids_counts = {tuple(pretoken.encode("utf-8")) : count for pretoken, count in pretoken_counts.items()}

    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    pair_to_ids = defaultdict(int) # キャッシュ
    
    
