from tokenizer import *
import io

""" pretokenizer """
# text = "hello hello hello world"
# for i in pretokenize(text):
#     print(i)

""" count_pairs """
# text = "hello"
# counts = count_pairs(list(text.encode("utf-8")), 3)
# print(counts)

""" merge """
# text = "hello"
# ids = list(text.encode("utf-8")) # [104, 101, 108, 108, 111]
# pair = (108, 108)
# new_id = 256

# merged_ids = merge(ids, pair, new_id) # [104, 101, 256, 111]
# print(merged_ids)

""" find_chunk_boundaries """
file_path = "japanese_codebot/dataset.txt"
end_token = "<|endoftext|>"
num_chunks = 10

# 1. ファイルの終端がb""で表せることを確認
# with open(file_path, "rb") as file:
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     # file.seek(0)
    
#     buffer_size = 256
    
#     buffer = file.read()
#     print(buffer)

# 2. end_tokenをfindした結果を確認
# byte_end_token = end_token.encode("utf-8")

# print(f"bytes end token: {byte_end_token}")

# with open(file_path, "rb") as file:
#     # print(f"読み込み文字コードを確認 : {io.TextIOWrapper(file, encoding="utf-8").encoding}")
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)
    
#     chunk_size = file_size // num_chunk
#     chunk_boundaries = [i * chunk_size for i in range(num_chunk)]
#     chunk_boundaries.append(file_size)
#     print(f"chunk size: {chunk_size}")
#     print(f"chunk boundaries : {chunk_boundaries}")
    
#     buffer_size = 256
    
#     for bi in range(1, len(chunk_boundaries) - 1):
#         print(f"境界値調整対象位置 : {chunk_boundaries[bi]}")
#         chunk_position = chunk_boundaries[bi]
#         file.seek(chunk_position)

#         while True:
#             buffer = file.read(buffer_size)
            
#             if buffer == b"":
#                 chunk_boundaries[bi] = file_size
#                 break
            
#             end_position = buffer.find(byte_end_token)
#             print(f"end token position : {end_position}")
            
#             if end_position != -1:
#                 chunk_boundaries[bi] = chunk_position + end_position
#                 file.seek(0)
#                 first_chunk = file.read(chunk_position + end_position)
#                 print(first_chunk.decode("utf-8", errors="replace"))
#                 break
            
#         break


# 3. 動作確認
# chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
# total_chunk = len(chunk_boundaries) - 1

# for i in range(total_chunk):
#     start = chunk_boundaries[i]
#     end = chunk_boundaries[i+1]
    
#     with open(file_path, "rb") as f:
#         f.seek(start)
#         chunk_byte = f.read(end - start)
#         chunk_text = chunk_byte.decode("utf-8", errors="ignore")
#         print(f""""読み込み開始""", start={start}, end={end}")
#         print(chunk_text)
