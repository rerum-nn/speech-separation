from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_images, plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["train" if self.is_train else "inference"]

        if self.modality == "audiovideo":
            video_features1 = self.video_encoder(batch["source1_mouth"].unsqueeze(1))
            video_features2 = self.video_encoder(batch["source2_mouth"].unsqueeze(1))
            batch["video_features"] = torch.cat(
                [video_features1.unsqueeze(1), video_features2.unsqueeze(1)], dim=1
            )

        with torch.cuda.amp.autocast(
            dtype=self.amp_dtype, enabled=self.use_amp or self.use_amp_bf16
        ):
            outputs = self.model(**batch)
            batch.update(outputs)

            batch = self._get_predicted(batch)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            loss = batch["loss"] / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()

        # update metrics for each loss (in case of multiple losses)
        if (
            not self.is_train
            or self._last_local_step % self._n_steps_update_metrics == 0
        ):
            for loss_name in self.config.writer.loss_names:
                metrics.update(loss_name, batch[loss_name].item())

            for met in metric_funcs:
                metrics.update(met.name, met(**batch))

        return batch

    def _get_predicted(self, batch):
        if "mask1" in batch and "mask2" in batch:
            batch["masked_spectrogram1"] = batch["mix_spectrogram"] * batch[
                "mask1"
            ].unsqueeze(1)
            batch["masked_spectrogram2"] = batch["mix_spectrogram"] * batch[
                "mask2"
            ].unsqueeze(1)
            batch["predicted_source1"] = self.audio_encoder.decode(
                batch["masked_spectrogram1"],
                batch["mix_phase"],
                batch["mix_waveform_len"],
                device=self.device,
            )
            batch["predicted_source2"] = self.audio_encoder.decode(
                batch["masked_spectrogram2"],
                batch["mix_phase"],
                batch["mix_waveform_len"],
                device=self.device,
            )
            batch["predicted"] = torch.cat(
                [batch["predicted_source1"], batch["predicted_source2"]], dim=1
            )
        elif "signal1" in batch and "signal2" in batch:
            batch["predicted_source1"] = batch["signal1"]
            batch["predicted_source2"] = batch["signal2"]
            batch["predicted"] = torch.cat(
                [batch["predicted_source1"], batch["predicted_source2"]], dim=1
            )
        elif "predicted" in batch:
            pass
        else:
            raise ValueError(f"Invalid model output. Batch keys: {batch.keys()}")

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        self.log_predictions(metric_funcs, **batch)

    def log_spectrogram(self, spectrogram, name):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image(name, image)

    def log_audio(
        self,
        mix_audio,
        predicted_audios,
        target_audio=None,
        aug_mix_audio=None,
        name="",
    ):
        if target_audio is not None:
            for i, target_audio in enumerate(target_audio):
                self.writer.add_audio(
                    f"{name}_target_audio_{i}",
                    target_audio.detach().cpu(),
                    sample_rate=self.sample_rate,
                )

        if aug_mix_audio is not None:
            for i, target_audio in enumerate(target_audio):
                self.writer.add_audio(
                    f"{name}_aug_mix_audio_{i}",
                    aug_mix_audio.detach().cpu(),
                    sample_rate=self.sample_rate,
                )

        for i, predicted_audio in enumerate(predicted_audios):
            self.writer.add_audio(
                f"{name}_predicted_audio_{i}",
                predicted_audio.detach().cpu(),
                sample_rate=self.sample_rate,
            )

        self.writer.add_audio(
            f"{name}_mix_audio", mix_audio.detach().cpu(), sample_rate=self.sample_rate
        )

    def log_images(self, images, subplots_names, name):
        images = images.detach().cpu()
        image = plot_images(images, subplots_names)
        self.writer.add_image(name, image)

    def log_predictions(self, metric_funcs, examples_to_log=2, **batch):
        rows = {}
        for i in range(min(examples_to_log, len(batch["mix_path"]))):
            name = Path(batch["mix_path"][i]).name.split(".")[0]

            self.log_spectrogram(batch["input_mix_spectrogram"][i], f"{name}_input_mix")
            self.log_spectrogram(
                batch["original_mix_spectrogram"][i], f"{name}_original_mix"
            )
            self.log_spectrogram(batch["mix_spectrogram"][i], f"{name}_mix")

            if "mask1" in batch and "mask2" in batch:
                self.log_spectrogram(batch["mask1"][i].unsqueeze(0), f"{name}_mask1")
                self.log_spectrogram(batch["mask2"][i].unsqueeze(0), f"{name}_mask2")
                self.log_spectrogram(
                    batch["masked_spectrogram1"][i], f"{name}_masked_spectrogram1"
                )
                self.log_spectrogram(
                    batch["masked_spectrogram2"][i], f"{name}_masked_spectrogram2"
                )

            self.log_audio(
                batch["original_mix"][i],
                batch["predicted"][i],
                batch["target"][i],
                batch["mix"][i] if batch["has_transforms"] else None,
                name,
            )

            if self.modality == "audiovideo":
                mouth1 = batch["source1_mouth"][i][0].unsqueeze(0)
                mouth2 = batch["source2_mouth"][i][0].unsqueeze(0)
                mouths = torch.cat([mouth1.unsqueeze(0), mouth2.unsqueeze(0)], dim=0)
                self.log_images(
                    mouths, ["source1_mouth", "source2_mouth"], f"{name}_mouths"
                )

            row = {"step": self.writer.step}

            for met in metric_funcs:
                row[met.name] = (
                    met(
                        predicted=batch["predicted"][i : i + 1],
                        target=batch["target"][i : i + 1],
                        mix=batch["mix"][i : i + 1],
                    )
                    .detach()
                    .cpu()
                    .item()
                )

            rows[name] = row

        if len(metric_funcs) > 0:
            self.writer.add_table(
                "metrics", pd.DataFrame.from_dict(rows, orient="index")
            )
