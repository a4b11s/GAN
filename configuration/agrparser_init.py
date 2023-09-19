import argparse


def agrparser_init():
    parser = argparse.ArgumentParser(
        prog='SA_WG_GAN',
        description='Self-attention WGAN'
    )

    parser.add_argument(
        '-hp',
        '--history_plot',
        action='store',
        type=str,
        dest="logfile_path",
        help="Plot training history from log file"
    )

    parser.add_argument(
        '-t',
        '--train',
        action='store_true',
        help="Start training"
    )

    parser.add_argument(
        '-sa',
        '--show_activation',
        action='store_true',
        help="Show layers activation"
    )

    return parser
