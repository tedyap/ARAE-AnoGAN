import argparse


def configure_args():
    parser = argparse.ArgumentParser(description='Training ARAE model')

    parser.add_argument('--model_dir', default='experiments/arae',
                        help="Directory containing params.json")
    parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
    parser.add_argument('--ckpts_dir', default='experiments/arae/ckpts',
                        help="Directory containing checkpoints")

    parser.add_argument('--vocab_size', type=int, default=9856, help='Vocabulary size')
    parser.add_argument('--max_len', type=int, default=30, help='Max length of sentence')

    return parser.parse_args()
