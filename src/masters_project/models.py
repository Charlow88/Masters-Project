import torch.nn as nn

OUTPUT_DIM = 16  # tau params length


def make_fc_model(output_dim: int = OUTPUT_DIM) -> nn.Module:
    """
    Fully-connected network:
      36 -> 720 -> 450 -> output_dim
    """
    model = nn.Sequential(
        nn.Linear(36, 720),
        nn.ReLU(),
        nn.Linear(720, 450),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(450, output_dim),
    )
    return model


def make_cnn_model(output_dim: int = OUTPUT_DIM) -> nn.Module:
    """
    CNN operating on the 6x6 measurement grid.
    """
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 6x6 -> 3x3

        nn.Flatten(),                           # 32 * 3 * 3 = 288
        nn.Linear(32 * 3 * 3, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, output_dim),
    )
    return model
