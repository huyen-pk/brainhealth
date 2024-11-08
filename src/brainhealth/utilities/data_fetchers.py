import tensorflow as tf
import numpy as np
from infrastructure.storage import Storage

# Fetch data from remote storage while training
class DataFetcher(tf.keras.utils.Sequence):
    def __init__(self, data_size:int, batch_size:int, storage: Storage):
        self.data_size = data_size
        self.batch_size = batch_size
        self.storage = storage

    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))
    
    def __getitem__(self, index):
        # Fetch data for the batch
        images, labels = self.storage.download(page_size=32, 
                                                page_index=index,
                                                page_count=1)

        return images, labels
    

# Serialize and fetch data from remote storage with TFRecord
class StaticDataFetcher:
    def __init__(self, data_size: int, batch_size: int, storage: Storage):
        self.data_size = data_size
        self.batch_size = batch_size
        self.storage = storage

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(self, image, label):
        feature = {
            'image': self._bytes_feature(image),
            'label': self._int64_feature(label),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def create_tfrecord(self, tfrecord_file):
        images, labels = self.storage.download(page_size=32, page_index=0)
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for image, label in zip(images, labels):
                tf_example = self.serialize_example(image, label)
                writer.write(tf_example)

    def load_tfrecord(self, tfrecord_file):
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }

        def parse_tfrecord_fn(example_proto):
            # Parse the input `tf.train.Example` proto using the feature description
            parsed_example = tf.io.parse_single_example(example_proto, feature_description)
            
            # Decode image
            image = tf.io.decode_jpeg(parsed_example['image'])
            
            # Convert label if needed
            label = tf.strings.to_number(parsed_example['label'], out_type=tf.int32)
            
            return image, label

        parsed_dataset = raw_dataset.map(parse_tfrecord_fn).batch(32)
        return parsed_dataset

