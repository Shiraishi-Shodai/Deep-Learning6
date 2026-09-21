import torch
import torch.nn.functional as F

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval() # 推論モード
    
    # プロンプトをトークン化
    device = next(model.parameters()).device # パラメータのデバイスを取得
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # 生成されたトークンを保持する変数
    generate_ids = ids.clone()

    # トークン生成ループ
    for _ in range(max_new_tokens):
        # コンテキスト長を超えた場合、古いトークンを切り捨てる
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]
        
        # 最後の位置のロジットを取得(次のトークン予測)
        logits = model(ids)[:, -1, :] # (1, V) = (B, V)
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            # 確率の高いN個を選ぶ
            next_id = torch.multinomial(probs, num_samples=1)

        # 終了トークンが生成されたら終了
        if next_id.item() == tokenizer.end_token_id:
            break
        
        # 生成したトークンを追加
        ids = torch.cat((ids, next_id), dim=1)
        generate_ids = torch.cat((generate_ids, next_id), dim=1)
    
    # デコードして返す(id list → bytes list → text)
    generate_text = tokenizer.decode(generate_ids[0].tolist())
    return generate_text