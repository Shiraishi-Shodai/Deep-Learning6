# UTF-8はUnicode文字をバイト列に変換するエンコード方式

# encoded = 'A'.encode("utf-8")
# print(encoded) # バイト列に変換
# print(list(encoded)) #各バイトの整数値を得る

# encoded = "あ".encode("utf-8")
# print(encoded)
# print(list(encoded))

# ids = [227, 129, 130]
# decoded = bytes(ids).decode("utf-8")
# print(decoded)

class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))
    
    def decode(self, ids):
        return bytes(ids).decode("utf-8")

tokenizer = ByteTokenizer()
text = "hello世界😆"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)
