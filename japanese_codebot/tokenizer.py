
# # トークナイザーを読み込み

# # テキストを読みこみ
# text_file = "japanese_codebot/dataset.txt"
# text = open(text_file).read()

# # テキストをトークンIDに変換(進捗バーを表示)

# # numpy配列に変換して保存

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
    
    return sorted(set(chunk_boundaries))