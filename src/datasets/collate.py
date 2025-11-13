import torch


def collate_list(items: list[dict], key: str) -> list:
    return [item[key] for item in items]


def collate_tensor(items, key: str) -> torch.Tensor:
    # suppose that all tensors have the same shape in the batch
    return torch.stack([item[key] for item in items])


def get_lengths(items: list[dict], key: str) -> torch.Tensor:
    return torch.tensor([item[key].shape[-1] for item in items], dtype=torch.long)


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = {
        "mix_path": collate_list(dataset_items, "mix_path"),
        "mix": collate_tensor(dataset_items, "mix"),
        "original_mix": collate_tensor(dataset_items, "original_mix"),
        "mix_len": dataset_items[0]["mix_len"],
        "mix_waveform_len": dataset_items[0]["mix"].shape[1],

        "mix_spectrogram": collate_tensor(dataset_items, "mix_spectrogram"),
        "mix_phase": collate_tensor(dataset_items, "mix_phase"),
        "input_mix_spectrogram": collate_tensor(dataset_items, "input_mix_spectrogram"),
        "original_mix_spectrogram": collate_tensor(dataset_items, "original_mix_spectrogram"),

        "has_transforms": dataset_items[0]["has_transforms"],
    }

    if "source1" in dataset_items[0]:
        batch["source1_path"] = collate_list(dataset_items, "source1_path")
        batch["source1"] = collate_tensor(dataset_items, "source1")

    if "source2" in dataset_items[0]:
        batch["source2_path"] = collate_list(dataset_items, "source2_path")
        batch["source2"] = collate_tensor(dataset_items, "source2")

    if "source1" in dataset_items[0] and "source2" in dataset_items[0]:
        batch["target"] = torch.cat([batch["source1"], batch["source2"]], dim=1)

    if "source1_mouth" in dataset_items[0]:
        batch["source1_mouth"] = collate_tensor(dataset_items, "source1_mouth")

    if "source2_mouth" in dataset_items[0]:
        batch["source2_mouth"] = collate_tensor(dataset_items, "source2_mouth")

    return batch
