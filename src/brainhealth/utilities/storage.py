import boto3
from botocore.exceptions import NoCredentialsError
import tempfile
import os
import numpy as np
from keras import preprocessing as pp
from abc import ABC, abstractmethod

class Storage(ABC):
    @abstractmethod
    def download(self, page_size, page_index, page_count=1, **kwargs):
        pass


class S3Storage(Storage):
    def __init__(self, bucket_name) -> None:
        self.bucket_name = bucket_name   
        self.s3 = boto3.client('s3')     

    def download(self, 
                page_size:int, 
                page_index:int, # starting page index to download
                page_count = 1,
                **kwargs) -> tuple[np.ndarray, np.ndarray]:
        bucket_name = self.bucket_name
        s3 = self.s3
        continuation_token = kwargs.get('continuation_token', None)
        folder_path = kwargs.get('folder_path', '')
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




if __name__ == '__main__':
    downloader = S3Storage('brainhealthtrainingdata')
    downloader.download(page_size=32, 
                                page_index=0,
                                page_count=1,
                                folder_path='test')
