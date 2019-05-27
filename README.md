# OOV Handling

This is thesis project for OOV Handling

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Requirements:

- [PyTorch 1.1](https://pytorch.org/)

- [polyglot](https://polyglot.readthedocs.io/en/latest/Installation.html)

- [nltk](https://www.nltk.org/install.html)

- [SciPy & Numpy](https://www.scipy.org/install.html)

Datasets:

- [Brown corpus from
  nltk](https://www.nltk.org/data.html)

- [word2vec pretrained
  embedding](https://code.google.com/archive/p/word2vec/)
  , put it under
  ```(project_folder)/embeddings/word_embedding/```,
  then run 
  ```
  python word2vec.py --max=[MAX_VOCAB] --local
  ```

- [dict2vec pretrained embedding](https://s3.us-east-2.amazonaws.com/dict2vec-data/dict2vec100.tar.bz2), put it 
  
- [polyglot english embedding](https://polyglot.readthedocs.io/en/latest/Download.html#langauge-task-support), under
  ```(project_folder)/embeddings/word_embedding/```
```
from polyglot.downloader import downloader
downloader.download("embeddings2.en")
```


### Training OOV model

To train the model:

```
python train.py --embedding=[polyglot|word2vec|dict2vec] --model=[lstm|cnn] --maxepoch=[MAX_EPOCH] --charlen=[CHARLEN] --lr=[LEARNING_RATE] --bsize=[BATCH_SIZE] --num_feature=[NUM_FEATURES] --charlen=[CHARLEN] --nesterov
```

To run train postagger:

```
python train_postag.py --continue_model --embedding=[polyglot|word2vec|dict2vec] --model=[lstm|cnn] --maxepoch=[MAXEPOCH] --charlen=[CHARLEN] --num_feature=[NUM_FEATURE]
```

To run word similarity test:
```
python --embedding=dict2vec --model=cnn --num_feature=[NUM_FEATURE]  --charlen=[CHARLEN] --local --load
```

make sure that ```num_feature``` and ```charlen``` is the same with the
trained model among run