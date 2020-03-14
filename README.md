# ARAE-AnoGAN for Text Anomaly Detection (Work In Progress)

Tensorflow 2.0 implementation of [`ARAE`](https://arxiv.org/pdf/1706.04223.pdf) and [`AnoGAN`](https://arxiv.org/pdf/1703.05921.pdf) to detect anomalies in text.

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

2. **Your first model** We created a `arae` directory for you under `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the model. It looks like
```angular2
{
    "batch_size": 64,
    "max_epoch": 60,
    "embedding_size": 300,
    "hidden_size": 300,
    "noise_size": 100,
    ...
}
```
For every new experiment, you will need to create a new directory under `experiments` with a `params.json` file.

3. **Train** your model. Simply run
```angular2
python train.py --model_dir experiments/<insert model directory name>
```
It will instantiate the model and train it on the training dataset with hyperparameters set in `params.json`. When training is done, you can check `output` directory under your model directory to look at samples of generated text.

4. **Custom dataset and hyperparameters search** To train the model with different dataset and perform hyperparameters search, run
```angular2
python train.py --data_dir <insert data directory>
```
and 
```angular2
python tune.py
```
You will need to modify `tune.py` to configure the process of hyperparameters search.