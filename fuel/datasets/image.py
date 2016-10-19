from glob import glob
from PIL import Image

import numpy

from fuel.datasets import Dataset
from fuel.schemes import SequentialExampleScheme


class ImagesFromFile(Dataset):
    """Creates a dataset from raw images on disk.

    Parameters
    ----------
    pattern : basestring
        The pattern which describes the path to all images to use. This is
        expected in form of a regular excursion. For example
        '/this/path/*jpeg' will use all jpeg images in the folder '/this/path'.
        The images will be given in source ``images`` and it will also
        give the source ``fiel_paths`` which contains the exact path of
        the loaded image.
    load_in_memory: bool, default=True
        If true, all images will be load in memory on construction.
        Otherwise the images are loaded as they are requested. When
        iterating multiple times over the dataset it is advicable to set it
        to true. If the dataset is very big and only a subset is used,
        it is advisable to put this to false.
    start: int, default=None
        If only a subset is required, the start of the block can be defined
        here. Note, the order is depended on the output of
        ``glob.glob(pattern)``.
    stop: int, default=None
        If only a subset is required, the end of the block can be defined
        here. Note, the order is depended on the output of
        ``glob.glob(pattern)``.

    """
    def __init__(self, pattern, load_in_memory=True, start=None, stop=None,
                 **kwargs):
        self.pattern = pattern
        self.load_in_memory = load_in_memory
        self.all_files = glob(pattern)[start:stop]

        # Load all images
        if load_in_memory:
            self.images = _load_images(self.all_files)

        # We give by default also the file path per image, in this way the
        # user can trace the image origin or can use a transformer to infer
        # the image label.
        self.provides_sources = ('images', 'file_paths')

        super(ImagesFromFile, self).__init__(**kwargs)

        self.example_iteration_scheme = SequentialExampleScheme(
            self.num_examples)

    @property
    def num_examples(self):
        return len(self.all_files)

    def get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError

        file_set = _list_fancy_indexing(self.all_files, request)
        if self.load_in_memory:
            image_set = _list_fancy_indexing(self.images, request)
        else:
            image_set = _load_images(file_set)
        data_dict = {'images': image_set, 'file_paths': file_set}
        return tuple(data_dict[source] for source in self.sources)


def _load_images(file_paths):
    images = []
    for fp in file_paths:
        img = Image.open(fp)
        images.append(numpy.array(img))
    return images


def _list_fancy_indexing(iterable, request):
    if isinstance(request, slice):
        return iterable[request]
    else:
        return [iterable[r] for r in request]
