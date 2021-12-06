import argparse
import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='corpus/pubmed/train.json')
    parser.add_argument('--save_model_dir', type=str, default='saved_model')
    parser.add_argument('--num_iter', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    trainer = Trainer.train(args)


if __name__ == "__main__":
    main()
