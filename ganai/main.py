import tensorflow as tf
import os

from ganai.configuration import Config, agrparser_init, ProgramArgs
from ganai.generate_image import generate_image
from ganai.plot_history import plot_history
from ganai.train import start_train


def start():
    argparser = agrparser_init()
    args: ProgramArgs = argparser.parse_args()

    print(tf.config.list_physical_devices("GPU"))

    config = Config()

    if args.logfile_path is not None:
        plot_history(args.logfile_path)
    elif args.generate:
        generate_image(args.batch_size, args.batch_count, config)
    elif args.train:
        start_train(args.epoch_count, config)


if __name__ == "__main__":
    start()
