import json
import logging
from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class AVSSDataset(BaseDataset):
    """
    Dataset for Audio-Visual Speech Separation (AVSS)

    Args:
        audio_dir (Path | str): Directory containing audio files with subdirectories:
            - mix/: Mixed audio files named as 'source1_id_source2_id.{wav,mp3,flac}'
            - s1/: Source 1 audio files (optional for inference)
            - s2/: Source 2 audio files (optional for inference)
        mouths_dir (Path | str): Directory containing mouth video files (.npz)
            named as 'source_id.npz'
    """

    def __init__(self, audio_dir: Path | str, mouths_dir: Path | str, *args, **kwargs):
        self.audio_dir = Path(audio_dir)
        self.mouths_dir = Path(mouths_dir)

        if not self.audio_dir.exists() or not self.mouths_dir.exists():
            raise ValueError("Invalid AVSSDataset directory")

        index = self._get_or_load_index()

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        """
        Load existing index from cache or create a new one.

        Returns:
            list[dict]: Index containing metadata for each dataset entry.
        """
        index_path = self.audio_dir / "index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        """
        Create index by scanning audio directory structure.

        Processes all mix files and looks for corresponding source audio files
        and mouth videos. At least one mouth video is required per entry.
        Sources are optional to support inference mode.

        Returns:
            list[dict]: Index containing metadata for each valid dataset entry.
        """
        data = []
        for path in Path(self.audio_dir / "mix").iterdir():
            try:
                if path.suffix not in [".mp3", ".wav", ".flac"]:
                    continue

                entry = {}
                entry["mix_path"] = str(path)
                t_info = torchaudio.info(entry["mix_path"])
                entry["mix_len"] = t_info.num_frames / t_info.sample_rate

                source1_id, source2_id = path.stem.split("_")
                entry["source1_id"] = source1_id
                entry["source2_id"] = source2_id

                source1_path = self.audio_dir / "s1" / path.name
                if source1_path.exists():
                    entry["source1_path"] = str(source1_path)
                    t_info = torchaudio.info(entry["source1_path"])
                    entry["source1_len"] = t_info.num_frames / t_info.sample_rate
                else:
                    logger.debug(
                        f"{path.name}: source1 file not found (inference mode)"
                    )

                source2_path = self.audio_dir / "s2" / path.name
                if source2_path.exists():
                    entry["source2_path"] = str(source2_path)
                    t_info = torchaudio.info(entry["source2_path"])
                    entry["source2_len"] = t_info.num_frames / t_info.sample_rate
                else:
                    logger.debug(
                        f"{path.name}: source2 file not found (inference mode)"
                    )

                source1_mouth_path = self.mouths_dir / (source1_id + ".npz")
                source2_mouth_path = self.mouths_dir / (source2_id + ".npz")

                if source1_mouth_path.exists():
                    entry["source1_mouth_path"] = str(source1_mouth_path)

                if source2_mouth_path.exists():
                    entry["source2_mouth_path"] = str(source2_mouth_path)

                if (
                    "source1_mouth_path" not in entry
                    and "source2_mouth_path" not in entry
                ):
                    logger.warning(
                        f"Skipping {path.name}: no mouth video found for either source"
                    )
                    continue

                data.append(entry)

            except Exception as e:
                logger.error(f"Error processing {path.name}: {e}")
                continue

        if len(data) == 0:
            logger.warning("No valid entries found in dataset")
        else:
            logger.info(f"Created index with {len(data)} entries")

        return data
