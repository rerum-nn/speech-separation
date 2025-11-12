from torch import nn
from torch.nn import Sequential


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, in_freq, in_frames, out_freq, out_frames, fc_hidden=512, *args, **kwargs):
        """
        Args:
            in_freq (int): number of input frequencies.
            in_frames (int): number of input frames.
            out_freq (int): number of output frequencies.
            out_frames (int): number of output frames.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=in_freq * in_frames, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
        )

        self.head1 = nn.Linear(in_features=fc_hidden, out_features=out_freq * out_frames)
        self.head2 = nn.Linear(in_features=fc_hidden, out_features=out_freq * out_frames)

    def forward(self, spectrogram, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
        Returns:
            output (dict): output dict containing mask1 and mask2.
        """
        output = self.net(spectrogram.view(spectrogram.size(0), -1))
        
        mask1 = self.head1(output)
        mask2 = self.head2(output)

        return {"mask1": mask1, "mask2": mask2}

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
