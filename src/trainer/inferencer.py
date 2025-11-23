import torch
import torchaudio
from tqdm.auto import tqdm
from pathlib import Path
from src.utils.audio_processing import match_rms

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        audio_encoder=None,
        video_encoder=None,
        sample_rate=16000,
        skip_model_load=False,
        rescale_audio=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            audio_encoder (AudioEncoder): audio encoder for the model.
            video_encoder (VideoEncoder): video encoder for the model.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.sample_rate = sample_rate

        self.modality = self.cfg_trainer.get("modality", "audio")
        if self.modality not in ["audio", "audiovideo"]:
            raise ValueError(f"Invalid modality: {self.modality}")

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(self.cfg_trainer.get("from_pretrained"))
        
        self.rescale_audio = rescale_audio

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def _get_predicted(self, batch):
        if 'mask1' in batch and 'mask2' in batch:
            batch['masked_spectrogram1'] = batch['mix_spectrogram'] * batch['mask1'].unsqueeze(1)
            batch['masked_spectrogram2'] = batch['mix_spectrogram'] * batch['mask2'].unsqueeze(1)
            batch['predicted_source1'] = self.audio_encoder.decode(batch['masked_spectrogram1'], batch['mix_phase'], batch['mix_waveform_len'], device=self.device)
            batch['predicted_source2'] = self.audio_encoder.decode(batch['masked_spectrogram2'], batch['mix_phase'], batch['mix_waveform_len'], device=self.device)
            batch['predicted'] = torch.cat([batch['predicted_source1'], batch['predicted_source2']], dim=1)
            
        return batch

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.modality == "audiovideo":
            video_features1 = self.video_encoder(batch['source1_mouth'].unsqueeze(1))
            video_features2 = self.video_encoder(batch['source2_mouth'].unsqueeze(1))
            batch['video_features'] = torch.cat([video_features1.unsqueeze(1), video_features2.unsqueeze(1)], dim=1)
        
        outputs = self.model(**batch)
        batch.update(outputs)

        batch = self._get_predicted(batch)

        if metrics is not None and "target" in batch:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["predicted"].shape[0]
        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            predicted = batch["predicted"][i].clone()

            if self.rescale_audio:
                source_1 = match_rms(batch["mix"][i], predicted[0]).unsqueeze(0).cpu()
                source_2 = match_rms(batch["mix"][i], predicted[1]).unsqueeze(0).cpu() 
            else:
                source_1 = predicted[0].unsqueeze(0).cpu()
                source_2 = predicted[1].unsqueeze(0).cpu() 
            
            mix_id = Path(batch["mix_path"][i]).stem

            mix_dir = self.save_path / part / "mix"
            s1_dir = self.save_path / part / "s1"
            s2_dir = self.save_path / part / "s2"

            mix_dir.mkdir(parents=True, exist_ok=True)
            s1_dir.mkdir(parents=True, exist_ok=True)
            s2_dir.mkdir(parents=True, exist_ok=True)

            mix_path = mix_dir / f"{mix_id}.wav"
            s1_path = s1_dir / f"{mix_id}.wav"
            s2_path = s2_dir / f"{mix_id}.wav"

            torchaudio.save(str(mix_path), batch["mix"][i].cpu(), self.sample_rate)
            torchaudio.save(str(s1_path), source_1, self.sample_rate)
            torchaudio.save(str(s2_path), source_2, self.sample_rate)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if "target" not in batch:
            print("Not target, skipping metrics calculation")

        return self.evaluation_metrics.result()
