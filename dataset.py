from io import BytesIO

import lmdb
from PIL import Image
from jittor.dataset import Dataset
import os


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        super().__init__()

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.total_len = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform


    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        #print(buffer)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


if __name__ == '__main__':
    from jittor import transform
    transforms = transform.Compose(
        [
            transform.RandomHorizontalFlip(),
            # transform.ToTensor(),
            transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


    def sample_data(dataset, batch_size, image_size=4):
        dataset.resolution = image_size
        # loader = dataset.set_attrs(shuffle=False, batch_size=batch_size, num_workers=0, drop_last=True)   
        # CHANGE BACK
        loader = dataset.set_attrs(shuffle=True, batch_size=batch_size, num_workers=2,
                                   drop_last=True)  # num_workers=1 
        return loader

    dataset = MultiResolutionDataset("lmdb/", transforms)
    loader = sample_data(dataset, batch_size=4)

    for i in range(1000):
        try:
            data = next(loader)
            print(i, data.shape)
        except:
            print(i, "error")
