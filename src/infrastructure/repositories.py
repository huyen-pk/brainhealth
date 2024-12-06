from botocore.exceptions import NoCredentialsError
import tempfile
import os
import numpy as np
import tensorflow as tf
from keras import preprocessing as pp
from keras import Model, models
from .storage import S3Storage
import tempfile
import datetime as dt
import shutil

class CheckpointRepository():
    def __init__(self, storage: S3Storage) -> None:
        self.storage = storage

    def get_latest(self, 
            model_name: str,
            **kwargs) -> str:
        """
        Load a model from a checkpoint directory.

        Parameters:
        model_name (str): The name of the model.
        checkpoint_dir (str): The directory where the model checkpoint was saved.

        Returns:
        str: Path to the saved checkpoint.
        """
        # Download the last 5 checkpoints from storage
        continuation_token = kwargs.get('continuation_token', None)
        # TODO: get the directory path from database
        s3prefix = self.get_checkpoint_remote_directory(model_name)
        directory = self.get_local_path(model_name)
        checkpoint_file=os.path.join(directory, 'checkpoint')
        try:
            self.storage.get(key=f"{s3prefix}/checkpoint", local_file_path=checkpoint_file)
        except Exception as e:
            print(f"Error downloading checkpoint file: {e}")
            return None
        if not os.path.exists(checkpoint_file) or os.path.getsize(checkpoint_file) == 0:
            return None
        checkpoints = None
        with open(checkpoint_file, "r") as file:
            for line in file:
                if(line.startswith("model_checkpoint_path")):
                    latest_checkpoint = line.split(":")[-1].strip()[1:-1]
                    checkpoints = self.storage.paging(
                                        save_to=directory,
                                        page_size=100,
                                        page_index=1,
                                        page_count=1,
                                        continuation_token=continuation_token,
                                        filter=f"{s3prefix}/{latest_checkpoint}")
                    break
        if checkpoints is None:
            return None
        shutil.copy(checkpoint_file, checkpoints)
        return tf.train.latest_checkpoint(checkpoints)
        
    def save(self, model_name: str, checkpoint: tf.train.Checkpoint):
        directory = self.get_local_path(model_name)
        file_prefix = self.get_checkpoint_file_prefix(model_name)
        return checkpoint.save(file_prefix=f"{directory}/{file_prefix}")
    
    
    def save_upload(self, model_name: str, checkpoint: tf.train.Checkpoint):
        # TODO: take into account the distributed training
        # Save the checkpoint to local drive
        local_path = self.save(model_name=model_name, checkpoint=checkpoint)

        # Upload the checkpoint to storage
        file_name = os.path.basename(local_path)
        directory = os.path.dirname(local_path)
        s3prefix = self.get_checkpoint_remote_directory(model_name)
        for file in os.listdir(directory):
            if file == 'checkpoint':
                file_path = os.path.join(directory, file)
                self.storage.save(file_path=file_path, prefix=f"{s3prefix}/{file}")
            if file_name in file:
                file_path = os.path.join(directory, file)
                self.storage.save(file_path=file_path, prefix=f"{s3prefix}/{file_name}/{file}")

    def get_checkpoint_remote_directory(self, model_name: str) -> str:
        return f'{os.getenv('CHECKPOINT_REPO_DIR_PATH')}/{model_name}'
    
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
        path = os.path.join(tmp_dir, "checkpoints", model_name)
        os.makedirs(path, exist_ok=True)
        return path

class ModelRepository():
    def __init__(self, storage: S3Storage) -> None:
        self.storage = storage
        
    def get(self, model_name: str, file_type: str) -> Model:
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
        model_file_path = os.path.join(self.get_local_path(model_name), f'{model_name}.{file_type}')
        if os.path.exists(model_file_path) and os.path.getsize(model_file_path) > 0:
            return self.load_from_file(model_file_path)
        s3prefix = f'{os.getenv('MODELS_REPO_DIR_PATH')}/{model_name}'
        key = f'{s3prefix}/{model_name}.{file_type}'
        self.storage.get(key=key, local_file_path=model_file_path)
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
        file_name = os.path.basename(file_path)
        s3prefix = f'{os.getenv('MODELS_REPO_DIR_PATH')}/{model_name}/{file_name}'
        try:
            # TODO: get the prefix(directory path) from database
            storage.save(file_path=file_path, prefix=s3prefix)
        except FileNotFoundError:
            print(f'The file was not found: {file_path}')
        except NoCredentialsError:
            print('Credentials not available')

    def get_local_path(self, model_name: str):
        local_repository = os.path.join(tempfile.gettempdir(), 'models', model_name)
        os.makedirs(local_repository, exist_ok=True)
        return local_repository
    
    def save_performance_metrics(self, epoch: int, model_name: str, metrics: dict, description: dict, identifier: str):
        file_path = os.path.join(self.get_local_path(model_name), f"performance_{dt.datetime.now().date()}_{identifier}.txt")
        with open(file_path, 'a') as file:
            file.write(f"Epoch {epoch} | ")
            for metric, value in metrics.items():
                file.write(f"{metric} = {np.float32(value):.6f} | ")
            file.write(f"Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Timestamp: {dt.datetime.now().timestamp()}\n")
        # TODO: take into account distributed training (file saved from multiple workers and in different timezones)
        # TODO: classify the file into periods (e.g. daily, weekly, monthly)
        s3prefix = f'{os.getenv('MODELS_PERF_REPO_DIR_PATH')}/{model_name}/{os.path.basename(file_path)}'
        self.storage.save(file_path=file_path, prefix=s3prefix)
    
class S3ImageDatasetRepository():
    def __init__(self, storage: S3Storage) -> None:
        self.storage = storage
    
    def get(self, 
            model_name: str,
            page_size:int, 
            page_index:int, # starting page index to download
            page_count = 1,
            **kwargs) -> tuple[np.ndarray, np.ndarray]:

        local_path = self.get_local_path(model_name, page_index)
        continuation_token = kwargs.get('continuation_token', None)
        s3_base_path = f'{os.getenv('TRAIN_DATA_DIR')}'
        s3prefixes = self.get_labels(model_name)
        for s3prefix in s3prefixes:
            filtering_key = f'{s3_base_path}/{s3prefix}'
            # TODO: stop when no more data is available for a label
            self.storage.paging(save_to=local_path,
                                page_size=page_size,
                                page_index=page_index,
                                page_count=page_count,
                                continuation_token=continuation_token,
                                filter=filtering_key)
        download_folder = os.path.join(local_path, s3_base_path)
        dataset = pp.image_dataset_from_directory(
            directory=download_folder,
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
        print(f"Local dataset storage: {local_storage}")
        os.makedirs(local_storage, exist_ok=True)
        return local_storage
    
    def get_labels(self, model_name: str) -> list[str]:
        # TODO: get the labels from database
        return os.getenv('TRAIN_DATA_LABELS', '').split(',')        
    
    