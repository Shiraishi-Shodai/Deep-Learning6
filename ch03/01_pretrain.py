import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append('.')

from itertools import cycle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from codebot.model import GPT
from codebot.utils import get_device

# 設定
device = get_device()
data_path = "codebot/tiny_codes.bin"
tokenizer_path = 'codebot/merge_rules.pkl'
model_save_path = 'codebot/model_pretrain.pt'

# ハイパーパラメータ
context_len = 256
vocab_size = 1000
batch_size = 32
learning_rate = 3e-4
max_iters = 20000
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = 4 * embed_dim
dropout_rate = 0.1

# データセットクラス
class TokenDataset(Dataset):
    def __init__(self, tokens, context_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len
    
    def __len__(self):
        return len(self.tokens) - self.context_len
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.context_len]
        y = self.tokens[idx+1:idx+self.context_len+1]
        return x, y

# データ準備
ids = np.fromfile(data_path, dtype=np.uint16)
dataset = TokenDataset(ids, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデル オプティマイザ
model = GPT(
    vocab_size=vocab_size,
    max_context_len=context_len,
    embed_dim=embed_dim,
    n_head=n_head,
    n_layer=n_layer,
    ff_dim=ff_dim,
    dropout_rate=dropout_rate
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# パラメータ数を計算
total_params = sum(p.numel() for p in model.parameters())
print(f"パラメータ数 : {total_params:,} ({total_params/1e6:.1f}M)")

losses = []
data_iter = cycle(dataloader) # 無限ループ
pbar = tqdm(range(max_iters))

x, y = next(data_iter)

for i in pbar:
    # (batch_size, context_len)のトークンIDを取り出す
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    
    # (batch_size, context_len, vocab_size)の確率を返す。
    # 各トークンに対する次のトークンの予測確率
    logits = model(batch_x)
    # 損失の計算(バッチサイズに左右されないようにデフォルトでlossを平均化している)
    # logits.view(-1, logits.size(-1)は(batch_size*context_len, vocab_size)に変形
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
    # 前回の勾配をクリア
    optimizer.zero_grad()
    # 逆伝搬
    loss.backward()
    # パラメータ更新
    optimizer.step()

    # 損失をlistに追加
    losses.append(loss.item())
    # 小数第4位までをpbarに表示
    pbar.set_postfix({'loss' : f'{loss.item():.4f}'})
    
# 結果を保存
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_pretrain.png')

model.save(model_save_path)