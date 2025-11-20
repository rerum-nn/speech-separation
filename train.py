import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import (
    load_video_encoder_weights,
    set_random_seed,
    setup_saving_and_logging,
)

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    audio_encoder = instantiate(config.audio_encoder)
    video_encoder = None

    if "video_encoder" in config:
        video_encoder = instantiate(config.video_encoder.model).to(device)
        video_encoder = load_video_encoder_weights(
            video_encoder, config.video_encoder.weights
        )

    sample_rate = config.trainer.sample_rate

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, audio_encoder, device)

    use_custom_init = config.trainer.get("use_custom_init", False)

    signal_length = dataloaders["train"].dataset[0]["mix"].shape[1]
    in_freq, in_frames = audio_encoder.get_input_shape(signal_length)
    out_freq, out_frames = audio_encoder.get_output_shape(signal_length)

    # build model architecture, then print to console
    model = instantiate(
        config.model,
        in_freq=in_freq,
        in_frames=in_frames,
        out_freq=out_freq,
        out_frames=out_frames,
        use_custom_init=use_custom_init,
    ).to(device)
    logger.info(model)

    use_pit = config.trainer.get("use_pit", True)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function, use_pit=use_pit).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(
                instantiate(metric_config, device=device, use_pit=use_pit)
            )

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        audio_encoder=audio_encoder,
        video_encoder=video_encoder,
        sample_rate=sample_rate,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
