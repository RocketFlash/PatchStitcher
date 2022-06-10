from torch.utils.data import Dataset as BaseDataset
from .utils import split_image_on_patches
from torchvision import transforms

class TilesDataset(BaseDataset):
    """Toles dataset
    
    Args:
        image     (np.array): image for tiles splitting
        window_size    (int): tile size
        step_size      (int): tiles step size
        is_horizontal (bool): orientation of tiles splitting
        transform (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
    
    """
    
    def __init__(
            self,
            image,
            window_size=20,
            step_size=10,
            is_horizontal=True,  
            is_multidirect=False, 
            transform=None):

        self.image = image
        self.tiles, self.tiler = split_image_on_patches(self.image, window_size=window_size,
                                                                    step_size=step_size,
                                                                    is_horizontal=is_horizontal,
                                                                    is_multidirect=is_multidirect)
        self.coords = self.tiler.crops 
            
        self.augmentation = transform
        
    
    def __getitem__(self, i):
        image=self.tiles[i]
        coords = self.coords[i]

        # apply augmentations
        if isinstance(self.augmentation, transforms.Compose):
            image = self.augmentation(image)
        else:
            sample = self.augmentation(image=image)
            image = sample['image']
            
        return image, coords
        
    def __len__(self):
        return len(self.tiles)