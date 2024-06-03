from activations_show import activations_show
from configuration.Config import Config
from configuration.agrparser_init import agrparser_init
from generate_image import generate_image
from plot_history import plot_history
from train import start_train

if __name__ == "__main__":
    argparser = agrparser_init()
    args = argparser.parse_args()

    config = Config()

    if args.logfile_path is not None:
        plot_history(args.logfile_path)
    elif args.generate:
        generate_image(args.batch_size, args.batch_count, config)
    elif args.show_activation:
        activations_show(config)
    elif args.train:
        start_train(args.epoch_count, config)