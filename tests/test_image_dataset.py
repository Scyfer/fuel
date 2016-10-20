import shutil
import tempfile
from PIL import Image

from fuel.datasets import ImagesFromFile
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme


def create_dummy_images(path, num_images):
    for i in range(num_images):
        im = Image.new('RGB', (512, 512), 0)
        im.save('{}/dummy{}.jpeg'.format(path, i))


class TestDataset(object):
    def setUp(self):
        self.ds_path = tempfile.mkdtemp()
        create_dummy_images(self.ds_path, 5)

    def tearDown(self):
        shutil.rmtree(self.ds_path)

    def test_num_examples(self):
        ds = ImagesFromFile('{}/*.jpeg'.format(self.ds_path))
        assert ds.num_examples == 5
        ds = ImagesFromFile('{}/*.jpeg'.format(self.ds_path), start=2)
        assert ds.num_examples == 3
        ds = ImagesFromFile('{}/*.jpeg'.format(self.ds_path), start=2, stop=3)
        assert ds.num_examples == 1

    def test_get_data_dynamic(self):
        ds = ImagesFromFile('{}/*.jpeg'.format(self.ds_path),
                            load_in_memory=False)
        stream = DataStream(ds, iteration_scheme=ShuffledScheme(
            ds.num_examples, batch_size=10))
        for imgs, _ in stream.get_epoch_iterator():
            assert len(imgs) == 5
            assert imgs[0].shape == (512, 512, 3)

    def test_get_data_inmemory(self):
        ds = ImagesFromFile('{}/*.jpeg'.format(self.ds_path),
                            load_in_memory=True)
        stream = DataStream(ds, iteration_scheme=ShuffledScheme(
            ds.num_examples, batch_size=10))
        for imgs, _ in stream.get_epoch_iterator():
            assert len(imgs) == 5
            assert imgs[0].shape == (512, 512, 3)
