from torch import nn
from src.model.rtfs_net_layers import AudioEncoder, AudioDecoder, RTFSBlock, VideoPreprocessor, CAF, S3MaskGenerator


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
            rtfs_blocks_num
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
            n_heads=n_heads
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
            *args, **kwargs):
        super().__init__()

        self.freq_dim = in_freq

        self.audio_encoder = AudioEncoder(hidden_dim)

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
            rtfs_blocks_num=rtfs_blocks_num
        )

        self.s3_mask_generator = S3MaskGenerator(hidden_dim)

        self.audio_decoder = AudioDecoder(hidden_dim)

    def forward(self, input_mix_spectrogram, mix_phase, video_features, *args, **kwargs):
        encoded_audio = self.audio_encoder(input_mix_spectrogram.permute(0, 1, 3, 2), mix_phase.permute(0, 1, 3, 2))
        encoded_video = video_features[:, 0, :, :].squeeze(1).permute(0, 2, 1)

        processed_audio = self.separation_network(encoded_audio, encoded_video)
        
        separated_audio = self.s3_mask_generator(processed_audio, encoded_audio)

        estimated_audio = self.audio_decoder(separated_audio)
        estimated_magnit = estimated_audio['magnit']
        estimated_phase = estimated_audio['phase']

        return {"magnit": estimated_magnit.permute(0, 1, 3, 2), "phase": estimated_phase.permute(0, 1, 3, 2)}

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
