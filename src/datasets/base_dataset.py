import logging
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        audio_encoder,
        target_sr=16000,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            audio_encoder (AudioEncoder): audio encoder for the model.
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.target_sr = target_sr
        self.instance_transforms = instance_transforms
        self.audio_encoder = audio_encoder

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]

        mix_path = data_dict["mix_path"]
        mix_len = data_dict["mix_len"]
        mix = self.load_audio(mix_path)

        instance_data = {
            "mix_path": mix_path,
            "mix_len": mix_len,
            "mix": mix,
            "original_mix": mix.clone()
            if self.instance_transforms is not None
            else mix,
            "has_transforms": self.instance_transforms is not None
            and "mix" in self.instance_transforms,
        }

        if "source1_path" in data_dict:
            instance_data["source1_path"] = data_dict["source1_path"]
            instance_data["source1_len"] = data_dict["source1_len"]
            instance_data["source1_id"] = data_dict["source1_id"]
            instance_data["source1"] = self.load_audio(data_dict["source1_path"])

        if "source2_path" in data_dict:
            instance_data["source2_path"] = data_dict["source2_path"]
            instance_data["source2_len"] = data_dict["source2_len"]
            instance_data["source2_id"] = data_dict["source2_id"]
            instance_data["source2"] = self.load_audio(data_dict["source2_path"])

        if "source1_mouth_path" in data_dict:
            instance_data["source1_mouth_path"] = data_dict["source1_mouth_path"]
            source1_mouth = self.load_video(data_dict["source1_mouth_path"])
            instance_data["source1_mouth"] = source1_mouth

        if "source2_mouth_path" in data_dict:
            instance_data["source2_mouth_path"] = data_dict["source2_mouth_path"]
            source2_mouth = self.load_video(data_dict["source2_mouth_path"])
            instance_data["source2_mouth"] = source2_mouth

        instance_data = self.preprocess_data(instance_data)

        mix_spectrogram, mix_phase = self.audio_encoder.encode(instance_data["mix"])
        instance_data["mix_spectrogram"] = mix_spectrogram
        instance_data["mix_phase"] = mix_phase

        original_mix_spectrogram = self.audio_encoder.get_spectrogram(
            instance_data["original_mix"]
        )
        instance_data["original_mix_spectrogram"] = original_mix_spectrogram

        input_mix_spectrogram = self.audio_encoder.encode_input(instance_data["mix"])
        instance_data["input_mix_spectrogram"] = input_mix_spectrogram

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[
            0:1, :
        ]  # TODO: а это зачем? он одноканальный, а если стерео не лучше ли среднее?
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.target_sr
            )
        return audio_tensor

    def load_video(self, video):
        return torch.from_numpy(np.load(video)["data"])

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "mix_path" in entry, (
                "Each dataset item should include field 'mix_path'"
                " - path to mix audio file."
            )
            assert "mix_len" in entry, (
                "Each dataset item should include field 'mix_len'"
                " - length of the mix audio."
            )
            assert "source1_mouth_path" in entry or "source2_mouth_path" in entry, (
                "Each dataset item should include at least one 'mouth_path'"
                " - path to mouth video file."
            )

    @staticmethod
    def _sort_index(index):
        """
        Sort index by audio length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["mix_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
