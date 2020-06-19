import torchvision as tv



augment = tv.transforms.RandomAffine(
    degrees=15,
    translate=(0.2, 0.2),
    scale=(0.9, 1.1),
    shear=10
)