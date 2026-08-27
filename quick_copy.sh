#!/bin/bash

basedir="/mnt/c/Users/siran/ML/deep-learning-from-scratch-6"
script_dir=$(cd "$(dirname "$0")" && pwd)
echo $script_dir

# ch始まりのディレクトリ取得
for dir in "$basedir"/ch*; do
    dirname=$(basename "$dir")
    # 指定したフォルダがない時だけフォルダを作成
    dirpath="${script_dir}/${dirname}"
    mkdir -p $dirpath
# pyファイルの一覧取得
    for file in "$dir"/*.py; do
    # pyファイル名の取得
        filename=$(basename "$file")
    # pyファイルのコピー(中身なしで)
        $(cp /dev/null "${script_dir}/${dirname}/${filename}") 
    done
done