from pathlib import Path

from src.datasets.avss_dataset import AVSSDataset


class CustomDirDataset(AVSSDataset):
    def __init__(self, path: Path | str, *args, **kwargs):
        path = Path(path)

        if not (path / "mouths").exists() or not (path / "audio").exists():
            raise ValueError("Invalid CustomDirDataset directory")

        audio_dir = path / "audio"
        mouths_dir = path / "mouths"

        super().__init__(audio_dir=audio_dir, mouths_dir=mouths_dir, *args, **kwargs)
