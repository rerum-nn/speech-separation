import torch
from torch import nn
from src.model.dprnn import DPRNN, DPRNNEncoder, DPRNNDecoder

class DPRNNTasNet(nn.Module):
    def __init__(
            self, 
            length,
            chunk_length,
            hop_length,
            hidden_dim=64, 
            dprnn_blocks=6,
            rnn_hidden_dim=128,
            kernel_size=3,
            stride=1,
            padding=1,
            *args,
            **kwargs):
        super().__init__()

        self.encoder = DPRNNEncoder(hidden_dim, kernel_size, stride, padding)
        self.dprnn = DPRNN(hidden_dim, length, chunk_length, hop_length, dprnn_blocks=dprnn_blocks, rnn_hidden_dim=rnn_hidden_dim)
        self.decoder = DPRNNDecoder(hidden_dim, kernel_size, stride, padding)

    def forward(self, mix, *args, **kwargs):
        embedding = self.encoder(mix)
        masks = self.dprnn(embedding)
        masked = masks * embedding.unsqueeze(1)
        predictions = [self.decoder(masked[:, i, :, :]) for i in range(masked.shape[1])]
        return {"signal1": predictions[0].unsqueeze(1), "signal2": predictions[1].unsqueeze(1)}

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
