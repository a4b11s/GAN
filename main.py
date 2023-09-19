from activations_show import activations_show
from configuration.agrparser_init import agrparser_init
from plot_history import plot_history
from train import start_train

if __name__ == "__main__":
    argparser = agrparser_init()
    args = argparser.parse_args()

    if args.logfile_path is not None:
        plot_history(args.logfile_path)
    if args.show_activation:
        activations_show()
    if args.train:
        start_train(args.epoch_count)
