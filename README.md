# Chinese NER OnnxRuntime C++

## Environment

```shell
conda create -n ner_onnx python==3.9
conda activate ner_onnx   
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Download Dataset

https://github.com/hspuppy/hugbert/blob/master/ner_dataset/msra.tgz

## Preprocess

```shell
mkdir -p dataset
tar -xzf msra.tgz -C dataset
python preprocess.py
```

## Train Model

```shell
bash run.sh
```