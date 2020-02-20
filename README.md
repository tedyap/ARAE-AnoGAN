# TextAnoGAN

Tensorflow 2.0 implementation of ARAE and AnoGAN to detect anomalies in text.

## Requirements

We recommend using python3 and a virtual env.

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Quickstart

1. **Get** [`reuters-21578`](http://www.daviddlewis.com/resources/testcollections/reuters21578/) dataset by running
```angular2
python get_data.py
```
It will write `reuters_train.txt` and `reuters_test.txt` under a new `data` directory. We will use these files to train and test our models.