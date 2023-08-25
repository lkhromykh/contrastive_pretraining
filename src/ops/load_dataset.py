import tensorflow as tf
import tensorflow_datasets as tfds
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass


def load_dataset(batch_size: int,
                 img_size: tuple[int, int]
                 ) -> tf.data.Dataset:
    ds = tfds.load('imagenet2012', split='all', as_supervised=True)

    def resize(img, label):
        img = tf.image.resize(img, img_size)
        return img, label
    ds = ds.shuffle(10000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(resize)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.as_numpy_iterator()
