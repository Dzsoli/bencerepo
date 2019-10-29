import unittest
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        data_file = "../../full_data/dataset.npy"
        labels_file = "../../full_data/labels.npy"
        self.dataset = np.load(data_file)
        self.labels = np.load(labels_file)

    def tearDown(self):
        del self.dataset
        del self.labels

    def test_dataset(self):
        self.assertEqual(self.labels.shape[0], self.dataset.shape[0])
        self.assertEqual(0, self.dataset.shape[0] % 3)


if __name__ == '__main__':
    unittest.main()
