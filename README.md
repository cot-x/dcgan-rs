# dcgan-rs
DCGAN via Rust. And EXE of Windows.

## dcgan-rs.exe --help
```
DCGAN via Rust.

Usage: dcgan-rs.exe [OPTIONS] --dataset <DATASET>

Options:
  -d, --dataset <DATASET>
  --batch-size <BATCH_SIZE>        [default: 32]
  --lr <LR>                        [default: 0.0001]
  --mul-dis-lr <MUL_DIS_LR>        [default: 4]
  --aug-threshold <AUG_THRESHOLD>  [default: 0.6]
  --aug-increment <AUG_INCREMENT>  [default: 0.01]
  -i, --iters <ITERS>                  [default: 1000000]
  -g, --generate <GENERATE>            [default: 0]
  -h, --help                           Print help
  -V, --version                        Print version
```

## How to use (Windows)
Releaseディレクトリ内にコンパイル済みの実行形式が保存されています。  
[PyTorch](https://pytorch.org/)から、LibTorchのC++ランタイムをダウンロードしてlibディレクトリ内のdllを同じディレクトリに入れてから実行して下さい。  
Datasetを[Kaggle](https://www.kaggle.com/datasets)や[Hugging Face](https://huggingface.co/datasets)や[Google](https://datasetsearch.research.google.com/)等からダウンロードしてお使い下さい。  
データセットの指定は、画像の入ったディレクトリ複数を含む親ディレクトリを指定して下さい。  
APAでデータオーギュメントしていて、数枚の画像セットから学習可能で、この様な小規模なセットならCPUでも現実的に実行可能です。  
また、大量のデータセットで学習を回す場合、Rustでの仕様上、メモリ上への読み込みに多少時間がかかります。  
