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


"""process_single_chunk"""
file_path = "japanese_codebot/dataset.txt"
end_token = "<|endoftext|>"
start = 0
end = 4960

# pretoken_counts = process_single_chunk(file_path, start, end, end_token)
# print(pretoken_counts)


"""pretoken_chunkc"""
# args = (file_path, start, end, end_token)
# pretoken_counts = pretoken_chunk(args)
# print(pretoken_counts)

"""train_bpe"""
file_path = "japanese_codebot/dataset.txt"
vocab_size = 500
end_token = "<|endoftext|>"
num_processes = 8
num_chunks = 8

merge_filepath = "japanese_codebot/merge_rules.pkl"

# if __name__ == "__main__":
#     merge_rules = train_bpe(file_path, vocab_size, end_token, num_processes, num_chunks)
    
#     with open(merge_filepath, "wb") as f:
#         pickle.dump(merge_rules, f)
    
"""BPE"""
if __name__ == "__main__":
    # tokenizer = BPETokenizer.load_from(merge_filepath)

    """_encode"""
    # counts = {
    #     (1, 2) : 3,
    #     (2, 1) : 2,
    #     (1, 5) : 1,
    #     (100, 10) : 2,
    # }
    
    # merge_rules = {
    #     (1, 2) : 3,
    #     (2, 1) : 2,
    #     (1, 5) : 10
    # }
    
    # def get_merge_priority(pair):
    #     return merge_rules.get(pair, float('inf'))  # 存在しないペアは最低優先度    
    
    # best_pair = min(counts, key=get_merge_priority)

    # print(best_pair)
    
    """encode"""
    input_text = "Hello" + end_token + "World"
    # pattern = '(' + re.escape(end_token) + ')'
    # texts = re.split(pattern, input_text)
    # print(texts)
    # print(input_text)
    
    # tokenizer = BPETokenizer.load_from(merge_filepath)
    # all_ids = tokenizer.encode(input_text)
    # print(all_ids)
    
    """_encode_chunk"""
    tokenizer = BPETokenizer.load_from(merge_filepath)