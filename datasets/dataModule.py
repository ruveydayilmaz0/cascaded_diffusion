import torch
from torch.utils.data import DataLoader, random_split
from datasets.HeLa_CTC import HeLaDataset
import glob
from pytorch_lightning import LightningDataModule


class DataModule(LightningDataModule):

    def __init__(
        self,
        batch_size,
        data_root,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_root = data_root
        # Create train/validation/test data splits
        self.set_data()

    def set_data(self):
        data_list = glob.glob(self.data_root + "/*")
        self.train_data, self.val_data, self.test_data = random_split(
            data_list, [0.70, 0.15, 0.15], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        if self.data_root is None:
            return None
        else:
            dataset = HeLaDataset(data_path=self.train_data, train=True)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )

    def test_dataloader(self):
        if self.data_root is None:
            return None
        else:
            dataset = HeLaDataset(data_path=self.test_data, train=True)
            return DataLoader(dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        if self.data_root is None:
            return None
        else:
            dataset = HeLaDataset(data_path=self.val_data, train=True)
            return DataLoader(dataset, batch_size=self.batch_size)
