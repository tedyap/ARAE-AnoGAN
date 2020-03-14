import argparse


def configure_args():
    parser = argparse.ArgumentParser(description='Training ARAE model')

    parser.add_argument('--model_dir', default='experiments/arae',
                        help="Directory containing params.json")
    parser.add_argument('--data_dir', default='data/reuters_short', help="Directory containing the dataset")

    parser.add_argument('--vocab_size', type=int, default=4158, help='Vocabulary size')
    parser.add_argument('--max_len', type=int, default=30, help='Max length of sentence')

    parser.add_argument('--anogan_epoch', type=int, default=100, help='max anogan epoch')


    parser.add_argument('--seed', type=int, default=500, help='random seed')
    parser.add_argument('--print_every', type=int, default=10, help='show metrics for train dataset')


    return parser.parse_args()
