import argparse
import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='corpus/pubmed/test0.json')
    parser.add_argument('--save_model_dir', type=str, default='saved_model')
    parser.add_argument('--num_iter', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()
    if args.mode == 'train':
        trainer = Trainer.train(args)
    elif args.mode == 'test':
        Trainer.test(args, 140)


if __name__ == "__main__":
    main()
