import tensorflow as tf
import tensorflow_datasets as tfds
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass
_N_SHUFFLE = 100


def _mixup(ds: tf.data.Dataset, lambda_: float) -> tf.data.Dataset:
    def convex(x, y, lam): return (1 - lam) * x + lam * y

    def mixup(batch0, batch1):
        lam = tf.random.uniform((), 0, lambda_)
        return convex(batch0[0], batch1[0], lam),\
               convex(batch0[1], batch1[1], lam)
    sds = ds.shuffle(_N_SHUFFLE)
    ds = tf.data.Dataset.zip(ds, sds)
    return ds.map(mixup)


def load_dataset(split: str,
                 *,
                 batch_size: int,
                 img_size: tuple[int, int],
                 mixup_lambda: float | None = None
                 ) -> tf.data.Dataset:
    ds, ds_info = tfds.load('imagenet2012', split=split,
                            as_supervised=True, with_info=True)

    def resize(img, label):
        img = tf.image.resize(img, img_size)
        label = tf.one_hot(label, ds_info.splits[split].num_classes)
        return img, label
    ds = ds.shuffle(_N_SHUFFLE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(resize)
    if mixup_lambda is not None:
        ds = _mixup(ds, mixup_lambda)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.as_numpy_iterator()
