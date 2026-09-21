import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')
import json
from codebot.tokenizer import BPETokenizer

# トークナイザの読み込み
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")

# JSONデータ読み込み
with open("codebot/tiny_codes_sft.json") as f:
    data = json.load(f)

# 1つ目のサンプルを取り出す
item = data[0]
print(item)
# {'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}

# Alpaca形式に変換
text = f"### Instruction:\n{item["instruction"]}\n\n### Response:\n{item["response"]}<|endoftext|>"
print(text)

### Instruction:
# Hello

### Response:
# Hello. What can I help you with?<|endoftext|>

# トークン化
ids = tokenizer.encode(text)