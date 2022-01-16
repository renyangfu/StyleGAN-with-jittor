import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from jittor import dataset
#from torchvision.transforms import functional as trans_fn
from jittor import transform


def resize_and_convert(img, size, quality=100):
    # img = transform.resize(img, size, Image.LANCZOS)
    # transform = transform.CenterCrop(size)
    # img = transform(img)
    transforms = transform.Compose([
        transform.Resize(size, Image.LANCZOS),
        transform.CenterCrop(size)
    ])
    img = transforms(img)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes=(8, 16, 32, 64, 128), quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))

    return imgs


def resize_worker(img_file, sizes):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes)

    return i, out


def prepare(transaction, dataset, n_worker, sizes=(8, 16, 32, 64, 128)):
    resize_fn = partial(resize_worker, sizes=sizes)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                transaction.put(key, img)

            total += 1

        transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    imgset = dataset.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=128 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, imgset, args.n_worker)
