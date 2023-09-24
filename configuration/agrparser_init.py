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
        '-e',
        '--epoch',
        action='store',
        type=int,
        dest="epoch_count",
        help="Start training",
        default=20
    )

    parser.add_argument(
        '-sa',
        '--show_activation',
        action='store_true',
        help="Show layers activation"
    )

    parser.add_argument(
        '-g',
        '--generate',
        action='store_true',
        help="Generate images"
    )

    parser.add_argument(
        '-bs',
        '--batch_size',
        action='store',
        help="Generate images batch_size",
        default=9,
        type=int
    )

    parser.add_argument(
        '-bc',
        '--batch_count',
        action='store',
        help="Generate images batch_count",
        default=1,
        type=int
    )

    return parser
