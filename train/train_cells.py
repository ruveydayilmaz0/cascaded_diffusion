# Adapted from https://github.com/lllyasviel/ControlNet

from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.dataset_CTC import CTCDataset
from cldm.model import create_model, load_state_dict
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse


# Configs
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train cells with cascaded diffusion model."
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Root for the dataset."
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="./models/control_sd15_ini_cells.ckpt",
        help="Path to the model checkpoint to resume from.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--logger_freq", type=int, default=300, help="Frequency of logging."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-6, help="Learning rate for training."
    )
    parser.add_argument(
        "--sd_locked",
        type=bool,
        default=False,
        help="Whether to lock the stable diffusion model.",
    )
    parser.add_argument(
        "--only_mid_control",
        type=bool,
        default=False,
        help="Whether to use only mid control.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of GPUs to train on (int) or which GPUs to \
            train on (list or str)",
    )
    return parser.parse_args()


def main(args):
    resume_path = args.resume_path
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    sd_locked = args.sd_locked
    only_mid_control = args.only_mid_control

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    experiment_name = "CTC_cells"
    wandb_logger = WandbLogger(name=experiment_name, project="cascaded")
    dataset = CTCDataset(args.dataset_root)
    dataloader = DataLoader(
        dataset, num_workers=32, batch_size=batch_size, shuffle=True
    )
    callbacks = [
        ModelCheckpoint(
            dirpath="ckpts/" + experiment_name,
            every_n_train_steps=100,
        ),
    ]
    trainer = pl.Trainer(
        gpus=args.gpus, precision=32, logger=wandb_logger, callbacks=callbacks
    )

    # Train!
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
