import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed, load_video_encoder_weights
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup audio and vidoe encoders
    audio_encoder = instantiate(config.audio_encoder)
    video_encoder = None

    if "video_encoder" in config:
        video_encoder = instantiate(config.video_encoder.model).to(device)
        video_encoder = load_video_encoder_weights(video_encoder, config.video_encoder.weights)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, audio_encoder, device)

    part = "test" if dataloaders.get("test") is not None else list(dataloaders.keys())[0]

    signal_length = dataloaders[part].dataset[0]["mix"].shape[1]
    in_freq, in_frames = audio_encoder.get_input_shape(signal_length)
    out_freq, out_frames = audio_encoder.get_output_shape(signal_length)

    # build model architecture, then print to console
    model = instantiate(config.model, in_freq=in_freq, in_frames=in_frames, out_freq=out_freq, out_frames=out_frames).to(device)
    print(model)

    # enable PIT inference for BSS models
    use_pit = config.inferencer.get("use_pit", False)

    # enable Rescale audio
    rescale_audio = config.inferencer.get("rescale_audio", False)

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        metrics["inference"].append(
            instantiate(metric_config, device=device, use_pit=use_pit)
        )

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        audio_encoder=audio_encoder,
        video_encoder=video_encoder,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
        rescale_audio=rescale_audio,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
