import torch



class SqueezeLayer(torch.nn.Module):
    def __init__(self, channels_in, channels_mid, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_mid, kernel_size, padding=padding),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(channels_mid, channels_in, kernel_size, padding=padding),
            torch.nn.LeakyReLU()
        )
    
    def forward(self, inputs):
        return inputs + self.layer(inputs)