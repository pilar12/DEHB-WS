from typing import Optional, Dict
import logging
import argparse

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms

from cnn import torchModel


def main(
    data_dir: str,
    max_epochs: int = 50,
    seed: int = 0,
    device: str = "cpu",
    cfg: Optional[Dict] = None,
):
    """
    the main function to run your AutoML System. It should only receive the basic information
    :param data_dir: str
        directory to where the dataset is stored
    :param seed: int
        random seeds
    :param device: str
        the device where the model runs
    :param cfg: Optional[Dict]
        hyperparameter configuration for the CNN to be evaluated. Default configurations will be evaluated if it is not
        given
    :return:
    """
    torch.manual_seed(seed)
    model_device = torch.device(device)

    img_width = 28
    img_height = 28
    input_shape = (1, img_width, img_height)

    pre_processing = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_val = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=pre_processing
    )

    test = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=pre_processing
    )
    if cfg is None:
        lr = 0.01
        batch_size = 100
        cfg = {
            "n_conv_layers": 3,
            "kernel_size": 3,
            "use_BN": True,
            "global_avg_pooling": False,
            "n_channels_conv_0": 16,
            "n_channels_conv_1": 32,
            "n_channels_conv_2": 64,
            "n_fc_layers": 0,
        }
    else:
        lr = cfg.get("learning_rate_init", 0.01)
        batch_size = cfg.get("batch_size", 100)

    model = torchModel(
        cfg, input_shape=input_shape, num_classes=len(train_val.classes)
    ).to(model_device)

    summary(model, input_shape, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_criterion = torch.nn.CrossEntropyLoss().to(model_device)

    train_loader = DataLoader(dataset=train_val, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    for epoch in range(max_epochs):
        logging.info("#" * 50)
        logging.info("Epoch [{}/{}]".format(epoch + 1, max_epochs))
        train_score, train_loss = model.train_fn(
            optimizer, train_criterion, train_loader, model_device
        )
        logging.info("Train accuracy %f", train_score)

    test_score = model.eval_fn(test_loader, device)
    print(f"Test accuracy {test_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAHS")
    parser.add_argument("--data_dir", type=str, default="./FashionMNIST")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="maximal number of epochs to train the network",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run the models"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    max_epochs = args.max_epochs
    seed = args.seed
    device = args.device

    main(data_dir=data_dir, max_epochs=max_epochs, seed=seed, device=device)
