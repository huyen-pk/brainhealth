import boto3
from botocore.exceptions import NoCredentialsError
import zipfile
import tempfile
import os
import numpy as np
from keras import preprocessing as pp

class S3downloader:
    def __init__(self, bucket_name) -> None:
        self.bucket_name = bucket_name        

    def download_from_s3(self, 
                        bucket_name:str,
                        page_size:int, 
                        page_index:int, # starting page index to download
                        page_count = 1,
                        continuation_token = None,
                        folder_path='') -> tuple[np.ndarray, np.ndarray]:
        s3 = boto3.client('s3')   
        local_path = ''
        paginator = s3.get_paginator('list_objects_v2')

        with tempfile.TemporaryDirectory(delete=False) as temp_dir:
            local_path = os.path.join(temp_dir, bucket_name)
            os.makedirs(local_path, exist_ok=True)
            try:
                for page_number in range(page_index + page_count):
                    page_iterator = paginator.paginate(
                                        Bucket=bucket_name, 
                                        Prefix=folder_path,
                                        PaginationConfig={
                                            'PageSize': page_size,
                                            'StartingToken': continuation_token
                                        })
                    # Skip pages until the desired page index
                    if page_number < page_index:
                        continue
                    page = page_iterator.build_full_result()
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        print("Downloading object: ", os.path.basename(key))
                        if not os.path.basename(key) == "":
                            local_file_path = os.path.join(local_path, key)
                            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                            s3.download_file(bucket_name, key, local_file_path)
                            print(f'Successfully downloaded {key} from bucket {bucket_name} to {local_file_path}')

                    if 'NextContinuationToken' in page_iterator:
                        continuation_token = page_iterator['NextContinuationToken']
                    else:
                        print('No more pages to download')
                        break
            except FileNotFoundError:
                print(f'The file was not found: {local_path}')
            except NoCredentialsError:
                print('Credentials not available')

        dataset = pp.image_dataset_from_directory(
            local_path,
            image_size=(32, 32),
            color_mode='rgb',
            batch_size=32,
            seed=123,
            shuffle=True
        )
        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)


import tensorflow as tf
import numpy as np

# Fetch data from remote storage while training
class DataFetcher(tf.keras.utils.Sequence):
    def __init__(self, data_size, batch_size):
        self.data_size = data_size
        self.batch_size = batch_size
        self.indices = np.arange(self.data_size)
        self.downloader = S3downloader()
    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))
    
    def __getitem__(self, index):
        # Fetch data for the batch
        images, labels = self.downloader.download_from_s3(page_size=32, 
                                                        page_index=index,
                                                        page_count=1,
                                                        folder_path='test')

        return images, labels

class StaticDataFetcher:
    def __init__(self, bucket_name, s3_key):
        self.bucket_name = bucket_name
        self.s3_key = s3_key
        self.downloader = S3downloader(bucket_name, s3_key)

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
        images, labels = self.downloader.download_from_s3(page_size=32, page_index=0)
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



if __name__ == '__main__':
    downloader = S3downloader('brainhealthtrainingdata')
    downloader.download_from_s3(page_size=32, 
                                page_index=0,
                                page_count=1,
                                folder_path='test')
