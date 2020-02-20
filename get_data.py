import argparse
import os
import string
import re
import nltk
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.corpus import stopwords
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")


def reuters_dataset(directory, train=True, test=True, clean_txt=True):
    """
    Load the Reuters-21578 dataset.
    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If true, load the training split of the dataset.
        test (bool, optional): If true, load the test split of the dataset.
        clean_txt (bool, optional): If true, clean text line (remove stopwords, punctuations, numbers, etc.)
    Returns:
        :class:`tuple` of dictionaries:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    nltk.download('reuters', download_dir=directory)
    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    doc_ids = reuters.fileids()

    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:

        split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))
        examples = []

        for id in split_set_doc_ids:
            if clean_txt:
                text = clean_text(reuters.raw(id))
            else:
                text = ' '.join(word_tokenize(reuters.raw(id)))
            labels = reuters.categories(id)

            examples.append({
                'text': text,
                'label': labels,
            })

        ret.append(examples)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def clean_text(text: str, rm_numbers=True, rm_punct=True, rm_stop_words=True, rm_short_words=True):
    """ Perform common NLP pre-processing tasks. """

    # make lowercase
    text = text.lower()

    # remove punctuation
    if rm_punct:
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # remove numbers
    if rm_numbers:
        text = re.sub(r'\d+', '', text)

    # remove whitespaces
    text = text.strip()

    # remove stopwords
    if rm_stop_words:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        text_list = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(text_list)

    # remove short words
    if rm_short_words:
        text_list = [w for w in text.split() if len(w) >= 3]
        text = ' '.join(text_list)

    return text


def generate_data_file(directory):
    normal_class = 0
    classes = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']

    # normal class: earn, the rest are anomalies
    normal_classes = [classes[normal_class]]
    del classes[normal_class]
    outlier_classes = classes
    train_set, test_set = reuters_dataset(directory=directory, train=True, test=True,
                                          clean_txt=False)
    train_idx_normal = []  # for subsetting train_set to normal class
    for i, row in enumerate(train_set):
        if any(label in normal_classes for label in row['label']) and (len(row['label']) == 1):
            train_idx_normal.append(i)
            row['label'] = 0
        else:
            row['label'] = 1
        row['text'] = clean_text(row['text'])

    test_idx = []  # for subsetting test_set to selected normal and anomalous classes
    label = []
    for i, row in enumerate(test_set):
        if any(label in normal_classes for label in row['label']) and (len(row['label']) == 1):
            test_idx.append(i)
            row['label'] = 0
            label.append(0)
        elif any(label in outlier_classes for label in row['label']) and (len(row['label']) == 1):
            test_idx.append(i)
            row['label'] = 1
            label.append(1)
        else:
            row['label'] = 2
        row['text'] = clean_text(row['text'])

    train_set_string = [train_set[i]["text"] for i in train_idx_normal]
    test_set_string = [test_set[i]["text"] for i in test_idx]

    with open(os.path.join(directory, 'reuters_train.txt'), 'w') as f:
        for i in train_set_string:
            f.write(i + '\n')

    with open(os.path.join(directory, 'reuters_test.txt'), 'w') as f:
        for i, line in enumerate(test_set_string):
            f.write(line + ',' + str(label[i]) + '\n')


if __name__ == '__main__':
    args = parser.parse_args()

    generate_data_file(args.data_dir)

