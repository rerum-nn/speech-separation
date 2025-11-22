import torch
from torch import nn
import torch.nn.functional as F
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
            rtfs_blocks_num,
            caf_blocks_num,
            caf_shared=True,
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

        self.caf_blocks_num = caf_blocks_num
        self.caf_shared = caf_shared

        if caf_shared:
            self.caf = CAF(C_v=video_dim, C_a=audio_dim, h=n_heads)
        else:
            self.caf = nn.ModuleList([CAF(C_v=video_dim, C_a=audio_dim, h=n_heads) for _ in range(caf_blocks_num)])

    def forward(self, a, v):
        a0 = a
        processed_audio = self.rtfs_block(a)
        processed_video = self.video_preprocessor(v)

        if self.caf_shared:
            for _ in range(self.caf_blocks_num):
                processed_audio = self.caf(processed_audio, processed_video)
        else:
            for caf in self.caf:
                processed_audio = caf(processed_audio, processed_video)

        for _ in range(self.rtfs_blocks_num):
            processed_audio = self.rtfs_block(processed_audio + a0)

        return processed_audio


class DecompressionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.transpose_conv(x)
        if skip.size(2) != x.size(2) or skip.size(3) != x.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='nearest')
        x = x + skip
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


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
            compression_blocks=0,
            n_heads=4,
            n_fft=256,
            win_length=256,
            hop_length=128,
            caf_blocks_num=1,
            caf_shared=True,
            *args, **kwargs):
        super().__init__()

        assert hidden_dim % (2 ** compression_blocks) == 0, "hidden_dim must be divisible by 2 ** compression_blocks"

        self.freq_dim = in_freq // (2 ** compression_blocks)

        self.compression_steps = compression_blocks
        self.compression_blocks = nn.ModuleList([])
        self.decompression_blocks = nn.ModuleList([])

        if compression_blocks > 0:
            cur_dim = hidden_dim // (2 ** (compression_blocks - 1))
        else:
            cur_dim = hidden_dim
        self.audio_encoder = AudioEncoder(cur_dim, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.audio_decoder = AudioDecoder(cur_dim, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.s3_mask_generator = S3MaskGenerator(cur_dim)

        for i in range(compression_blocks):
            if i == 0:
                self.compression_blocks.append(
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(in_channels=cur_dim, out_channels=cur_dim, kernel_size=3, padding=1),
                        nn.ReLU()
                    )
                )
            else:
                self.compression_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=cur_dim // 2, out_channels=cur_dim, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=cur_dim, out_channels=cur_dim, kernel_size=3, padding=1),
                        nn.ReLU()
                    )
                )

            if i == compression_blocks - 1:
                self.decompression_blocks.insert(0, DecompressionBlock(in_channels=cur_dim, out_channels=cur_dim))
            else:
                self.decompression_blocks.insert(0, DecompressionBlock(in_channels=cur_dim * 2, out_channels=cur_dim))

            cur_dim = cur_dim * 2

        self.separation_network = RTFSSeparationNetwork(
            processing_dim=processing_dim, 
            audio_dim=hidden_dim, 
            video_dim=video_encoder_dim, 
            freq_dim=self.freq_dim, 
            rnn_layers=rnn_layers, 
            rnn_hidden_dim=rnn_hidden_dim, 
            vp_compression_blocks=vp_compression_blocks, 
            ap_compression_blocks=ap_compression_blocks, 
            n_heads=n_heads, 
            rtfs_blocks_num=rtfs_blocks_num,
            caf_blocks_num=caf_blocks_num,
            caf_shared=caf_shared,
        )

    def forward(self, mix, video_features, *args, **kwargs):
        encoded_audio = self.audio_encoder(mix)

        skips = []
        encoded_compressed_audio = encoded_audio
        for i in range(self.compression_steps):
            encoded_compressed_audio = self.compression_blocks[i](encoded_compressed_audio)
            skips.append(encoded_compressed_audio)
            encoded_compressed_audio = F.max_pool2d(encoded_compressed_audio, kernel_size=2, stride=2)

        encoded_videos = [video_features[:, i, :, :].squeeze(1).permute(0, 2, 1) for i in range(video_features.shape[1])]

        estimated_audios = []
        for encoded_video in encoded_videos:
            processed_audio = self.separation_network(encoded_compressed_audio, encoded_video)
            
            for i in range(self.compression_steps):
                skip = skips[self.compression_steps - i - 1] 
                processed_audio = self.decompression_blocks[i](processed_audio, skip)

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
