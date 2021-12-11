import argparse
import glob
import logging
import math
import os

from skimage.io import imread, imsave
from skimage.transform import SimilarityTransform, warp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from api import PRN
from render import FaceRenderer

logger = logging.getLogger(__name__)

class FaceDataset(Dataset):
    def __init__(self, face_images, expand_bb=1.6, cropped_res=256):
        self.face_images = face_images
        self.expand_bb = expand_bb
        self.cropped_res = cropped_res

    def __len__(self):
        return len(self.face_images)

    def __getitem__(self, index):
        image, bb, rel_path = self.face_images[index]
        image = image/255.
        h, w, c = image.shape
        if bb is not None:
            bb[0] -= bb[1] * (self.expand_bb - 1) / 2
            bb[1] *= self.expand_bb
        else:
            # in case we have invalid bonding box info
            bb = np.array([[0.,0.],[w,h]])
        scale = math.sqrt(self.cropped_res ** 2 / np.prod(bb[1]))
        tform = SimilarityTransform(
            scale=scale,
            translation=self.cropped_res / 2 - (bb[0] + bb[1] / 2) * scale,
        )
        image = warp(image, tform.inverse, output_shape=(self.cropped_res, self.cropped_res))
        return image, os.path.splitext(rel_path)[0]

class CelebA:
    def __init__(self, path):
        self.base_path = path
        def f():
            exts = ('.jpg', '.png')
            for e in exts:
                yield from glob.iglob(os.path.join(path, '*/*/live/*' + e))
        self.image_path_list = list(tqdm(f(), desc='Enumerating images'))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        p = self.image_path_list[index]
        image = imread(p)
        h, w, c = image.shape
        p_bb = os.path.splitext(p)[0] + '_BB.txt'
        try:
            # [[x0, y0], [w, h]]
            bb = np.array([float(b) for b in open(p_bb).read().split()[:4]]).reshape(2,2)
            bb /= 224.
            bb[:,0] *= w
            bb[:,1] *= h
        except IndexError:
            # in case we have invalid bonding box info
            bb = None
        return image, bb, os.path.relpath(p, self.base_path)

def main(args):
    dataloader = DataLoader(
        FaceDataset(
            CelebA(args.celebA_path),
        ),
        batch_size=64,
        num_workers=32,
    )
    prn = PRN(is_dlib=False)

    triangles = prn.face_ind[prn.triangles]
    r = FaceRenderer(triangles)
    for images, paths in tqdm(dataloader):
        pos = prn.pos_predictor.predict_batch(images.numpy())
        rendered_imgs = r.draw_batch(pos)

        m = rendered_imgs.min(axis=(1,2), keepdims=True)
        rendered_imgs = (1 - rendered_imgs) / (1 - m)

        for img, p in zip(rendered_imgs, paths):
            p = os.path.join(args.output, p) + '.png'
            try:
                os.makedirs(os.path.dirname(p))
            except OSError:
                pass
            imsave(p, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--celebA_path', default='/mnt/cephfs/dataset/FAS/CelebA_Spoof/CelebA_Spoof/Data')
    parser.add_argument('--output', default='/mnt/cephfs/dataset/FAS/CelebA_Spoof/depth')
    args = parser.parse_args()
    main(args)

    # ds = CelebA(args.celebA_path)
    # t = time.time()
    # for d in ds:
    #     t2 = time.time()
    #     print(d[2], '%.5f' % (t2 - t))
    #     t = t2
