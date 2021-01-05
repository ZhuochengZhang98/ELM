# A ELM sentiment classifier

This is a sentiment classifier implemented via **E**xtreme **L**earning **m**achine and Pretrained Language Model.

## 1. Requirements

### For ELM
- numpy==1.19.4
- matplotlib==3.3.3(optional)
- tqdm==4.54.0(optional)
- sklearn==0.23.2(optional)

### For sentiment classification
- pytorch==1.7.0
- transformers==4.0.0
- matplotlib==3.3.3
- pandas==1.1.4
- tqdm==4.54.0

## 2. Getting Started

### 2.1 Using ELM
There are 3 types of elm in elm.py:
- basic_elm: basic implementation of elm(single layer, binary classification only)
- normal_elm: normalize with a parameter before calculating Moore–Penrose inverse(single layer, binary classification only)
- classic_elm(**recommend**): single layer elm for multi-classes

The detailed usage can be found in elm_example.py

You can also use the wrapper ELM from elm.py.
```python
from elm import ELM
from argparse import ArgumentParser

def main():
    # parse args
    parser = ArgumentParser()
    parser.add_argument('--type', type=str, default='classic')
    parser.add_argument('--input_shape', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--normalize', action=float, default=1.0)
    parser.add_argument('--classes', type=int, default=2)
    args = parser.parse_args()
    
    # load ELM
    elm = ELM(args)
```

### 2.2 Using sentiment classification
To run sentiment classification task simply run sentiment.py in your command line.
```bash
python sentiment.py --training_type finetune_classifier_elm \
 --batch_size 64 \
 --epoch_num 6 \
 --learning_rate 1e-5 \
 --eval_epoch 1
```
For detailed usage, run:
```bash
python sentiment.py --help
```

## 3. TODO
- [ ] add rbf kernel for elm
- [ ] add multi-layer elm
- [ ] add chinese dataset support for sentiment classify