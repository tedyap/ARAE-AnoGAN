# ARAE-AnoGAN for Text Anomaly Detection

[Illinois Wesleyan University Research Honors] The final paper is presented [here](https://digitalcommons.iwu.edu/cgi/viewcontent.cgi?article=1023&context=cs_honproj).

## Abstract
Generative adversarial networks (GANs) are now one of the
key techniques for detecting anomalies in images, yielding remarkable
results. Applying similar methods to discrete structures, such as text sequences, is still largely an unknown. In this work, we introduce a new
GAN-based text anomaly detection method, called ARAE-AnoGAN,
that trains an adversarially regularized autoencoder (ARAE) to reconstruct normal sentences and detects anomalies via a combined anomaly
score based on the building blocks of ARAE. Finally, we present experimental results demonstrating the effectiveness of ARAE-AnoGAN and
other deep learning methods in text anomaly detection.

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