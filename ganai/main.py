import click

from ganai.tune import tune
from ganai.utilites import ConfigLoader
from ganai.train import train
from ganai.generate_image import generate_image


@click.group()
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True),
    help="Config file path",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    if config_path is None:
        config_path = "./ganai/config.yml"

    config_fields = [
        "img_size",
        "noise_dim",
        "kid_image_size",
        "g_filters_start",
        "g_filters_multiplayer",
        "g_attentions",
        "d_filters_start",
        "d_filters_multiplayer",
        "d_attentions",
    ]

    train_config = ConfigLoader(config_fields=config_fields, yml_path=config_path)
    ctx.ensure_object(dict)
    ctx.obj["config"] = train_config.config


@click.command(name="train", help="Train the model")
@click.argument("chp_path", type=click.Path(exists=True, dir_okay=True))
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
    default=False,
    help="Verbose mode",
)
@click.pass_context
def start_train(
    ctx: click.Context,
    chp_path: str,
    epochs: int,
    batch_size: int,
    verbose: bool,
) -> None:
    config = ctx.obj["config"]

    train(
        epochs=epochs,
        batch_size=batch_size,
        chp_path=chp_path,
        verbose=verbose,
        model_config=config,
    )


@click.command(name="tune", help="Tune HP of the model")
@click.option("-e", "--epochs", type=int, default=100, help="Number of epochs")
@click.option(
    "-b",
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for training",
)
@click.pass_context
def start_tuning(
    ctx: click.Context,
    epochs: int,
    batch_size: int,
) -> None:
    config = ctx.obj["config"]

    tune(
        epochs=epochs,
        batch_size=batch_size,
        model_config=config,
    )


@click.command(name="generate", help="Generate images")
@click.argument("chp_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(exists=True, dir_okay=True))
@click.option(
    "-bs", "--batch_size", type=int, default=9, help="Generate images batch_size"
)
@click.option(
    "-bc", "--batch_count", type=int, default=1, help="Generate images batch_count"
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose mode")
@click.pass_context
def start_generate(
    ctx: click.Context,
    chp_path: str,
    output_path: str,
    batch_size: int,
    batch_count: int,
    verbose: bool,
) -> None:
    config = ctx.obj["config"]

    generate_image(
        chp_path=chp_path,
        output_path=output_path,
        batch_size=batch_size,
        batch_count=batch_count,
        verbose=verbose,
        model_config=config,
    )


def main() -> None:
    cli.add_command(start_train)
    cli.add_command(start_generate)
    cli.add_command(start_tuning)
    cli()


if __name__ == "__main__":
    main()
