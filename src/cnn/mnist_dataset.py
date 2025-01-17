import os
import struct
import numpy as np
import gzip
import os
import shutil
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf



def load_MNIST(dataset="training", path='../MNIST/'):
    if dataset is "training":
        f_img = os.path.join(path, 'train-images.idx3-ubyte')
        f_labels = os.path.join(path, 'train-labels.idx1-ubyte')
    else:
        f_img = os.path.join(path, 't10k-images.idx3-ubyte')
        f_labels = os.path.join(path, 't10k-labels.idx1-ubyte')

    with open(f_labels, 'rb') as flabels:
        magic, num = struct.unpack(">II", flabels.read(8))
        labels = np.fromfile(flabels, dtype=np.int8)

    with open(f_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        imgs = np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), rows, cols)
    return imgs, labels

#imgs_train, labels_train = load_MNIST('training')
#imgs_test, labels_test = load_MNIST('testing')


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
    if magic != 2049:
        raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
      tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def dataset(images_file, labels_file):
    """Download and parse MNIST dataset."""

    #images_file = download(directory, images_file)
    #labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(
      images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train():
    """tf.data.Dataset object for MNIST training data."""
    return dataset('../../MNIST/train-images.idx3-ubyte',
                 '../../MNIST/train-labels.idx1-ubyte')


def test():
    """tf.data.Dataset object for MNIST test data."""
    return dataset('../../MNIST/t10k-images.idx3-ubyte',
                 '../../MNIST/t10k-labels.idx1-ubyte')

if __name__ == "__main__":
    a = train()
    print(a)
