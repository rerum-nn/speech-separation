import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def load_audio(path, target_sr):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor


def get_audio_files(folder: Path, exts=(".wav", ".mp3", ".flac")):
    files = {}
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            files.setdefault(p.stem, p)
    return files


def load_mix_wavs_directory(directory_path, target_sr):
    directory = Path(directory_path)
    mix_dir = directory / "mix"

    if not mix_dir.is_dir():
        raise ValueError(f"Expected mix folder in {directory}")

    mix_files = get_audio_files(mix_dir)

    wavs = {}
    for id in mix_files.keys():
        wavs[id] = load_audio(mix_files[id], target_sr)

    return wavs


def load_sources_wavs_directory(directory_path, target_sr):
    directory = Path(directory_path)
    s1_dir = directory / "s1"
    s2_dir = directory / "s2"

    if not s1_dir.is_dir() or not s2_dir.is_dir():
        raise ValueError(f"Expected s1 and s2 folders in {directory}")

    s1_files = get_audio_files(s1_dir)
    s2_files = get_audio_files(s2_dir)

    common_ids = set(s1_files.keys()) & set(s2_files.keys())
    if not common_ids:
        raise ValueError(f"No common audio files found in {s1_dir} and {s2_dir}")

    wavs = {}
    for id in common_ids:
        s1 = load_audio(s1_files[id], target_sr)
        s2 = load_audio(s2_files[id], target_sr)

        wavs[id] = torch.cat([s1, s2], dim=0).unsqueeze(0)

    return wavs


def calculate_metrics(config: DictConfig):
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    print(f"using device: {device}")

    target_sr = config.get("sample_rate", 16000)
    print(f"sample rate: {target_sr}")

    use_pit = config.get("use_pit", False)

    mix_wavs = load_mix_wavs_directory(config.paths.mix, target_sr)
    ground_truth_wavs = load_sources_wavs_directory(
        config.paths.ground_truth, target_sr
    )
    predicted_wavs = load_sources_wavs_directory(config.paths.predictions, target_sr)

    common_ids = (
        set(mix_wavs.keys())
        & set(ground_truth_wavs.keys())
        & set(predicted_wavs.keys())
    )

    if not common_ids:
        raise ValueError("no common ids")

    print(f"found {len(common_ids)} common ids")

    metrics = []
    for metric_config in config.metrics:
        metric = instantiate(metric_config, device=device, use_pit=use_pit)
        metrics.append(metric)

    print(f"metrics: {[m.name for m in metrics]}")

    metric_totals = {metric.name: 0.0 for metric in metrics}
    count = 0

    for id in tqdm(common_ids):
        mix = mix_wavs[id].to(device)
        ground_truth = ground_truth_wavs[id].to(device)
        predicted = predicted_wavs[id].to(device)

        batch = {
            "mix": mix,
            "predicted": predicted,
            "target": ground_truth,
        }

        for metric in metrics:
            metric_value = metric(**batch)
            metric_totals[metric.name] += metric_value

        count += 1

    results = {"num_wavs": count}

    for metric_name, total_value in metric_totals.items():
        avg_value = total_value / count if count > 0 else 0.0
        results[metric_name] = avg_value

    return results


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config: DictConfig):
    results = calculate_metrics(config)
    for key, value in results.items():
        if key != "num_wavs":
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
