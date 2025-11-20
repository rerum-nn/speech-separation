import torch
from torch import nn

from src.model.rtfs_net_layers import (
    CAF,
    AudioDecoder,
    AudioEncoder,
    RTFSBlock,
    S3MaskGenerator,
    VideoPreprocessor,
)
from src.model.rtfs_net_layers.custom_init import init_weights


class RTFSSeparationNetwork(nn.Module):
    def __init__(
        self,
        audio_dim,
        video_dim,
        processing_dim,
        freq_dim,
        rnn_layers,
        rnn_hidden_dim,
        vp_compression_blocks,
        ap_compression_blocks,
        n_heads,
        rtfs_blocks_num,
    ):
        super().__init__()

        self.rtfs_blocks_num = rtfs_blocks_num

        self.rtfs_block = RTFSBlock(
            audio_dim,
            processing_dim,
            freq_dim,
            rnn_layers=rnn_layers,
            rnn_hidden_dim=rnn_hidden_dim,
            compression_blocks=ap_compression_blocks,
            n_heads=n_heads,
        )
        self.video_preprocessor = VideoPreprocessor(
            video_dim,
            processing_dim,
            compression_blocks=vp_compression_blocks,
            compression_kernel_size=3,
        )

        self.caf = CAF(C_v=video_dim, C_a=audio_dim, h=n_heads)

    def forward(self, a, v):
        a0 = a
        processed_audio = self.rtfs_block(a)
        processed_video = self.video_preprocessor(v)

        processed_audio = self.caf(processed_audio, processed_video)
        for _ in range(self.rtfs_blocks_num):
            processed_audio = self.rtfs_block(processed_audio + a0)

        return processed_audio


class RTFSNet(nn.Module):
    def __init__(
        self,
        in_freq,
        hidden_dim=256,
        processing_dim=64,
        rtfs_blocks_num=12,
        video_encoder_dim=512,
        rnn_layers=4,
        rnn_hidden_dim=32,
        vp_compression_blocks=4,
        ap_compression_blocks=2,
        n_heads=4,
        n_fft=256,
        win_length=256,
        hop_length=128,
        use_custom_init=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.freq_dim = in_freq

        self.audio_encoder = AudioEncoder(
            hidden_dim,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            custom_init=use_custom_init,
        )

        self.separation_network = RTFSSeparationNetwork(
            processing_dim=processing_dim,
            audio_dim=hidden_dim,
            video_dim=video_encoder_dim,
            freq_dim=in_freq,
            rnn_layers=rnn_layers,
            rnn_hidden_dim=rnn_hidden_dim,
            vp_compression_blocks=vp_compression_blocks,
            ap_compression_blocks=ap_compression_blocks,
            n_heads=n_heads,
            rtfs_blocks_num=rtfs_blocks_num,
        )

        self.s3_mask_generator = S3MaskGenerator(hidden_dim)

        self.audio_decoder = AudioDecoder(
            hidden_dim,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            custom_init=use_custom_init,
        )

        if use_custom_init:
            self.apply(init_weights)

    def forward(self, mix, video_features, *args, **kwargs):
        encoded_audio = self.audio_encoder(mix)
        encoded_videos = [
            video_features[:, i, :, :].squeeze(1).permute(0, 2, 1)
            for i in range(video_features.shape[1])
        ]

        estimated_audios = []
        for encoded_video in encoded_videos:
            processed_audio = self.separation_network(encoded_audio, encoded_video)

            separated_audio = self.s3_mask_generator(processed_audio, encoded_audio)

            estimated_audio = self.audio_decoder(separated_audio)
            estimated_audios.append(estimated_audio)

        estimated_audios = torch.cat(estimated_audios, dim=1)

        return {"predicted": estimated_audios}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
