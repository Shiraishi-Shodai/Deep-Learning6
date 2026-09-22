import regex

pattern = r"""
    \p{Hiragana}+        # ひらがな
  | \p{Katakana}+        # カタカナ
  | \p{Han}+             # 漢字
  | \p{Latin}+           # ラテン文字
  | \p{N}+               # 数字
  | [^\p{Hiragana}\p{Katakana}\p{Han}\p{Latin}\p{N}\s]+
                           # 記号
  | \s+                  # 空白
"""

def pretokenize(text):
    return regex.findall(pattern, text, regex.VERBOSE)

pretoken = pretokenize("私はPythonでLLMを作っています。")
print(pretoken)