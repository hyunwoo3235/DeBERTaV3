import tensorflow as tf


def _parse_function(example_proto):
    features = {"text": tf.io.FixedLenFeature([512], tf.int64)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features["text"]


# make tfrecord dataset iterable from files
def TFRecordDataset(
    filenames,
    compression_type=None,
    buffer_size=None,
    shuffle=2048,
    batch_size=32,
    drop_remainder=True,
    num_parallel_reads=tf.data.experimental.AUTOTUNE,
    prefetch=tf.data.experimental.AUTOTUNE,
):
    dataset = tf.data.TFRecordDataset(filenames, compression_type, buffer_size)
    dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel_reads)
    if shuffle > 0:
        dataset = dataset.shuffle(shuffle)

    dataset = dataset.batch(batch_size, drop_remainder).prefetch(prefetch)
    return dataset
