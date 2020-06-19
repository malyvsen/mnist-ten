import torch



class SqueezeLayer(torch.nn.Module):
    def __init__(self, channels_in, channels_mid, kernel_size=3):
        super().__init__()
        self.short = torch.nn.AvgPool2d(2)
        padding = kernel_size // 2
        self.long = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_mid, kernel_size, padding=padding),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(channels_mid, channels_in, kernel_size, padding=padding),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2)
        )
    
    def forward(self, inputs):
        return self.short(inputs) + self.long(inputs)