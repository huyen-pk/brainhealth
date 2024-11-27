from botocore.exceptions import NoCredentialsError
import tempfile
import os
import numpy as np
import tensorflow as tf
from keras import preprocessing as pp
from keras import Model, models
from .storage import S3Storage
import tempfile

class CheckpointRepository():
    def __init__(self, storage: S3Storage) -> None:
        self.storage = storage

    def get_latest(self, 
            model_name: str,
            **kwargs) -> str:
        """
        Load a model from a checkpoint directory.

        Parameters:
        checkpoint_dir (str): The directory where the model checkpoint is saved.

        Returns:
        str: Path to the saved checkpoint.
        """
        # Download the last 5 checkpoints from storage
        continuation_token = kwargs.get('continuation_token', None)
        # TODO: get the directory path from database
        checkpoint_dir = f'{os.getenv('CHECKPOINT_DIR')}/{model_name}'
        checkpoints = self.storage.paging(page_size=5,
                              page_index=1,
                              page_count=1,
                              continuation_token=continuation_token,
                              folder_path=checkpoint_dir)
        return tf.train.latest_checkpoint(checkpoints)
        
    def save(self, model_name: str, checkpoint: tf.train.Checkpoint):
        directory = self.get_local_path(model_name)
        file_prefix = self.get_checkpoint_file_prefix(model_name)
        return checkpoint.save(file_prefix=f"{directory}/{file_prefix}")
    
    
    def save_upload(self, model_name: str, checkpoint: tf.train.Checkpoint):
        local_path = self.save(model_name=model_name, checkpoint=checkpoint)
        self.storage.save(file_path=local_path, prefix=model_name)
    
    def get_checkpoint_file_prefix(self, model_name: str) -> str:
        return f"{model_name}_chkpt"

    def get_local_path(self, model_name: str) -> tuple[str, str]:
        """
        Get the path to the checkpoint directory on local drive.

        Parameters:
        model_name (str): The name of the model.

        Returns:
        str: The path to the checkpoint directory.
        """
        tmp_dir = tempfile.gettempdir()
        path = os.makedirs(os.path.join(tmp_dir, "checkpoints", model_name), exist_ok=True)
        return path

class ModelRepository():
    def __init__(self, storage: S3Storage) -> None:
        self.storage = storage
        
    def get(self, model_name: str) -> Model:
        """
        Download a model from a url.

        Parameters:
        model_url (str): The url to download the model file.

        Returns:
        str: Local path to downloaded file.
        """
        # TODO: get the filename and suffix to download from database, for now let's assume the model is .h5
        # TODO: exception handling
        # TODO: load from cache if available or refresh cache
        model_file_path = os.path.join(self.get_local_path(model_name), f'{model_name}.h5')
        self.storage.get(model_name, model_file_path)
        return self.load_from_file(model_file_path)
    
    def load_from_file(self, model_file_path: str) -> Model:
        """
        Load a pre-trained model from a file path.

        Parameters:
        model_file_path (str): The file path to the pre-trained model.

        Returns:
        tf.keras.Model: The pre-trained model.
        """
        if model_file_path is None:
            raise ValueError('Model file path is required.')
        
        if not os.path.exists(model_file_path) or os.path.getsize(model_file_path) == 0:
                raise FileNotFoundError(f'Model not found at {model_file_path}')
        
        return models.load_model(model_file_path, compile=False)

    def save(self, model_name:str, file_path: str):
        storage = self.storage
        bucket_name = self.bucket_name
        try:
            # TODO: get the prefix(directory path) from database
            storage.save(file_path=file_path, prefix=model_name)
            print(f'Successfully uploaded {file_path} to {bucket_name}')
        except FileNotFoundError:
            print(f'The file was not found: {file_path}')
        except NoCredentialsError:
            print('Credentials not available')

    def get_local_path(self, model_name: str):
        local_repository = os.path.join(tempfile.gettempdir(), 'models', model_name)
        os.makedirs(local_repository, exist_ok=True)
        return local_repository
    
    def save_performance_metrics(self, epoch: int, model_name: str, metrics: dict, description: dict):
        file_path = os.path.join(self.get_local_path(model_name), "performance.txt")
        with open(file_path, 'a') as file:
            for metric, value in metrics.items():
                file.write(f"Epoch {epoch}: {metric} = {value} | Description: {description[metric]}\n")
    
class S3ImageDatasetRepository():
    def __init__(self, storage: S3Storage) -> None:
        self.storage = storage
    
    def get(self, 
            model_name: str,
            page_size:int, 
            page_index:int, # starting page index to download
            page_count = 1,
            **kwargs) -> tuple[np.ndarray, np.ndarray]:

        save_to = self.get_local_path(model_name, page_index)
        continuation_token = kwargs.get('continuation_token', None)
        local_path = self.storage.paging(save_to=save_to,
                                page_size=page_size,
                                page_index=page_index,
                                page_count=page_count,
                                continuation_token=continuation_token,
                                folder_path=model_name)

        dataset = pp.image_dataset_from_directory(
            local_path,
            image_size=(32, 32),
            color_mode='rgb',
            batch_size=page_size,
            seed=123,
            shuffle=True
        )
        images = []
        labels = []
        for image, label in dataset: # iterate over each batch of data
            images.append(image.numpy()) # add one batch to the list
            labels.append(label.numpy())
        return np.array(images), np.array(labels)

    def get_local_path(self, model_name: str, batch_index: int):
        local_storage = os.path.join(tempfile.gettempdir(), 'datasets', model_name, f'batch_{batch_index}')
        os.makedirs(local_storage, exist_ok=True)
        return local_storage
    
    