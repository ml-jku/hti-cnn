import numpy as np
from tensorpack.dataflow import *


class ILSVRC12Files(RNGDataFlow):
    """
    Same as :class:`ILSVRC12`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """

    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Same as in :class:`ILSVRC12`.
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        assert meta_dir is None or os.path.isdir(meta_dir), meta_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        if name == 'train':
            dir_structure = 'train'
        if dir_structure is None:
            dir_structure = ILSVRCMeta.guess_dir_structure(self.full_dir)

        meta = ILSVRCMeta(meta_dir)
        self.imglist = meta.get_image_list(name, dir_structure)

        for fname, _ in self.imglist[:10]:
            fname = os.path.join(self.full_dir, fname)
            assert os.path.isfile(fname), fname

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            fname = os.path.join(self.full_dir, fname)
            yield [fname, label]
