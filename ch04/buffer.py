import os

file_path = "storybot/tiny_stories_train.txt"

with open(file_path, "rb") as file:
    print(type(file))
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    print(file_size)
    file.seek(0)
    print(file.tell())
        
text = "Hello world<|endoftext|>Hello Python<|endoftext|>Hello AI."
end_token = "<|endoftext|>".encode("utf-8")
text_bytes = text.encode("utf-8")

print(end_token)
print(text_bytes)
print(text_bytes.find(end_token))
