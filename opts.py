import argparse


def configure_args():
    parser = argparse.ArgumentParser(description='Training ARAE model')

    parser.add_argument('--model_dir', default='experiments/arae',
                        help="Directory containing params.json")
    parser.add_argument('--data_dir', default='data/reuters_short_constant', help="Directory containing the dataset")
    parser.add_argument('--ckpts_dir', default='experiments/arae/ckpts',
                        help="Directory containing checkpoints")

    parser.add_argument('--vocab_size', type=int, default=4389, help='Vocabulary size')
    #parser.add_argument('--vocab_size', type=int, default=11000, help='Vocabulary size')
    parser.add_argument('--max_len', type=int, default=30, help='Max length of sentence')
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--print_every', type=int, default=10, help='show metrics for train dataset')


    return parser.parse_args()
