import PIL
import glob
from skimage import io
from torch.utils.data import Dataset
import glob
from torchvision import transforms

class CTCMaskDataset(Dataset):
    """
    Dataset of fluorescently labeled cell nuclei
    """

    def __init__(
        self,
        paths,
        max_val = 0.5,
        min_val = 0.0
    ):
        self.paths = paths
        self.max_val = max_val
        self.min_val = min_val
        self.mask_paths = []

        for sub_dir in sorted(glob.glob(self.paths + 'masks/*/')):
            self.mask_paths.extend(sorted(glob.glob(sub_dir + '*.tif')))
        
        self.augmentations = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        batch = {}
        mask_filename = self.mask_paths[idx]
        mask = io.imread(mask_filename)
        mask = (mask - self.min_val) / (self.max_val - self.min_val)
        mask = 2*mask - 1.0
        mask = self.augmentations(PIL.Image.fromarray(mask))
        batch["input"] = mask
        return batch