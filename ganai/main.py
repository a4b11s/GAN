import click

from ganai.configuration import Config
from ganai.train import train


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    pass


@click.command(name="train", help="Train the model")
@click.argument("chp_path", type=click.Path(exists=True, dir_okay=True), help="Checkpoint path")
@click.option("-e", "--epochs", type=int, default=100, help="Number of epochs")
@click.option(
    "-b",
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for training",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=True,
    help="Verbose mode",
)
def start_train(
    chp_path: str,
    epochs: int,
    batch_size: int,
    verbose: bool,
) -> None:
    config = Config()
    train(
        epochs=epochs,
        batch_size=batch_size,
        chp_path=chp_path,
        verbose=verbose,
        config=config,
    )


def main() -> None:
    cli.add_command(start_train)
    cli()


if __name__ == "__main__":
    main()
