import boto3
from botocore.exceptions import NoCredentialsError
import tempfile
import os
import numpy as np
from keras import preprocessing as pp
from abc import ABC, abstractmethod
import json
class BlobStorage(ABC):
    @abstractmethod
    def get(self, key: str, local_file_path: str) -> str:
        pass

    @abstractmethod
    def save(self, file_path: str, prefix: str):
        pass

    @abstractmethod
    def paging(self, 
                page_size:int, 
                page_index:int, # starting page index to download
                page_count = 1,
                **kwargs) -> tuple[np.ndarray, np.ndarray]:
        pass
class S3Storage(BlobStorage):
    def __init__(self, connection_string: str) -> None:
        connection_info = json.loads(connection_string)
        bucket_name = connection_info['bucket_name']
        access_key = connection_info['access_key']
        access_secret = connection_info['access_secret']
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=access_secret
        )
        self.bucket_name = bucket_name
        
    def get(self, key: str, local_file_path: str) -> str:
        bucket_name = self.bucket_name
        self.s3.download_file(Bucket=bucket_name, Key=key, Filename=local_file_path)
        return local_file_path

    def paging(self, 
                save_to: str,
                page_size:int, 
                page_index:int, # starting page index to download
                page_count = 1,
                **kwargs) -> tuple[np.ndarray, np.ndarray]:
        s3 = self.s3
        bucket_name = self.bucket_name 
        continuation_token = kwargs.get('continuation_token', None)
        folder_path = kwargs.get('folder_path', '')
        paginator = s3.get_paginator('list_objects_v2')

        os.makedirs(save_to, exist_ok=True)
        try:
            for page_number in range(page_index + page_count):
                page_iterator = paginator.paginate(
                                    Bucket=bucket_name, 
                                    Prefix=folder_path, # model_name: training data for the given model
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
                        local_file_path = os.path.join(save_to, key)
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        s3.download_file(Bucket=bucket_name, Key=key, Filename=local_file_path)
                        print(f'Successfully downloaded {key} from bucket {bucket_name} to {local_file_path}')
                if 'NextContinuationToken' in page_iterator:
                    continuation_token = page_iterator['NextContinuationToken']
                else:
                    print('No more pages to download')
                    break
        except FileNotFoundError:
            print(f'The file was not found: {save_to}')
        except NoCredentialsError:
            print('Credentials not available')

        return save_to

    def save(self, file_path: str, prefix: str):
        """
        Save file to blob storage.
        
        Parameters:
        file_path (str): path to the file to be uploaded on local drive.
        prefix (str): prefix which represents folder structure in blob storage.
        """

        s3 = self.s3
        bucket_name = self.bucket_name
        try:
            s3.upload_file(Filename=file_path, Bucket=bucket_name, Key=prefix)
            print(f'Successfully uploaded {file_path} to {bucket_name}/{prefix}')
        except FileNotFoundError:
            print(f'The file was not found: {file_path}')
        except NoCredentialsError:
            print('Credentials not available')


class LocalStorage(BlobStorage):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.all_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.all_files.append(os.path.join(root, file))

        # Sort files to ensure consistent paging
        self.all_files.sort()
        self.start_index = 0

    def get_dataset(self, page_size, page_index, page_count=1, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        
        all_files = self.all_files
        self.start_index = page_index * page_size
        end_index = self.start_index + (page_size * page_count)
        paged_files = all_files[self.start_index:end_index]

        temp_dir = tempfile.mkdtemp()
        for file_path in paged_files:
            relative_path = os.path.relpath(file_path, self.data_dir)
            dest_path = os.path.join(temp_dir, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            os.symlink(file_path, dest_path)

        dataset = pp.image_dataset_from_directory(
            temp_dir,
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