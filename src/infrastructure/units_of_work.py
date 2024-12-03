import os
import datetime as dt
from abc import ABC, abstractmethod
from typing import override
from brainhealth.models.conf import VariableNames
from infrastructure.repositories import ModelRepository, CheckpointRepository, S3ImageDatasetRepository
import tensorflow as tf
from keras import Model
import numpy as np

class ModelTrainingDataDomain(ABC):
    def __init__(self, 
                 model_repository: ModelRepository, 
                 checkpoint_repository: CheckpointRepository,
                 dataset_repository: S3ImageDatasetRepository) -> None:
        self.__model_repository__ = model_repository
        self.__checkpoint_repository__ = checkpoint_repository
        self.__dataset_repository__ = dataset_repository

    def get_model(self, model_name: str, file_type="h5") -> Model:
        return self.__model_repository__.get(model_name=model_name, file_type=file_type)

    def save_model(self, model_name: str, file_path: str) -> None:
        self.__model_repository__.save(model_name=model_name, file_path=file_path)
    
    def save_model(self, model_name: str, model: Model, type: str = 'h5') -> str:
        repo = self.get_model_repository_local(model_name)
        file_path = os.path.join(repo, f"{model_name}_{str(dt.datetime.now().timestamp())}.{type}")
        model.save(file_path)
        self.__model_repository__.save(model_name=model_name, file_path=file_path)    

    def save_weights(self, model_name: str, model: Model) -> str:
        repo = self.get_model_repository_local(model_name)
        file_path = os.path.join(repo, f"{model_name}_{str(dt.datetime.now().timestamp())}.weights.h5")
        model.save_weights(file_path)
        self.__model_repository__.save(model_name=model_name, file_path=file_path)
        return file_path
    
    def get_model_repository_local(self, model_name: str) -> str:
        return self.__model_repository__.get_local_path(model_name)
    
    def get_label_classes(self, model_name: str) -> list[str]:
        return self.__dataset_repository__.get_labels(model_name)
    
    def get_dataset(self, model_name: str, page_size: int, page_index: int, page_count=1, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self.__dataset_repository__.get(model_name=model_name, 
                                           page_size=page_size, 
                                           page_index=page_index, 
                                           page_count=page_count,
                                           **kwargs)
    
    def purge_dataset(self, model_name: str, batch_index: int) -> None:
        local_path = self.__dataset_repository__.get_local_path(model_name, batch_index)
        if os.path.exists(local_path):
            shutil.rmtree(local_path)

    def get_latest_checkpoint(self, model_name) -> str:
        """
        Get the latest checkpoint for a model.

        Parameters:
        model_name (str): The name of the model.

        Returns:
        str: The path on local drive to the latest checkpoint.
        """
        return self.__checkpoint_repository__.get_latest(model_name)
    
    def get_checkpoint_local_path(self, model_name: str) -> str:
        return self.__checkpoint_repository__.get_local_path(model_name)

    def save_checkpoint(self, model_name: str, checkpoint: tf.train.Checkpoint) -> None:
        self.__checkpoint_repository__.save_upload(model_name=model_name, checkpoint=checkpoint)

    def save_performance_metrics(self, epoch:int, model_name: str, metrics: dict, descriptions: dict) -> None:
        self.__model_repository__.save_performance_metrics(epoch=epoch, model_name=model_name, metrics=metrics, description=descriptions)

import numpy as np
from keras import models, preprocessing as pp
import tempfile
import datetime as dt
import shutil
class Local_ModelTrainingDataDomain(ModelTrainingDataDomain):
    def __init__(self, 
                 model_repository: ModelRepository = None, 
                 checkpoint_repository: CheckpointRepository = None,
                 dataset_repository: S3ImageDatasetRepository = None) -> None:
        self.model_repository = model_repository
        self.checkpoint_repository = checkpoint_repository
        self.dataset_repository = dataset_repository
        self.all_files = []
        self.dataset_repo = os.getenv(VariableNames.TRAIN_DATA_DIR)
        for root, dirs, files in os.walk(self.dataset_repo):
            for file in files:
                try:
                    self.all_files.append(os.path.join(root, file))
                except Exception as e:
                    print(f"Error with file {file}: {e}")
        # Sort files to ensure consistent paging
        self.all_files.sort()
        self.start_index = 0


    @override
    def get_model(self, model_name: str) -> Model:
        model_repo = self.get_model_repository_local(model_name)
        model_file_path = os.path.join(model_repo, model_name + '.h5')
        if not os.path.exists(model_file_path) or os.path.getsize(model_file_path) == 0:
                raise FileNotFoundError(f'Model not found at {model_file_path}')
        
        return models.load_model(model_file_path, compile=False)

    @override
    def save_model(self, file_path: str, model_name: str):
        pass

    @override
    def get_model_repository_local(self, model_name: str) -> str:
        local_repository = os.getenv(VariableNames.MODELS_REPO_DIR_PATH)
        model_path = os.path.join(local_repository, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        return model_path
    
    @override
    def get_dataset(self, page_size, page_index, page_count=1, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        dataset_repo = self.dataset_repo
        if not os.path.exists(dataset_repo) or not os.listdir(dataset_repo):
            raise FileNotFoundError(f'Dataset repository not found or is empty at {dataset_repo}')
        
        all_files = self.all_files
        self.start_index = page_index * page_size
        end_index = self.start_index + (page_size * page_count)
        paged_files = all_files[self.start_index:end_index]

        temp_dir = tempfile.mkdtemp()
        for file_path in paged_files:
            relative_path = os.path.relpath(file_path, dataset_repo)
            dest_path = os.path.join(temp_dir, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            os.symlink(file_path, dest_path)

        dataset = pp.image_dataset_from_directory(
            dataset_repo,
            image_size=(32, 32),
            color_mode='rgb',
            batch_size=page_size,
            seed=123,
            shuffle=True
        )

        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)
    
    @override
    def purge_dataset(self, model_name: str, batch_index: int) -> None:
        pass

    @override
    def get_latest_checkpoint(self, model_name) -> str:
        """
        Get the latest checkpoint for a model.

        Parameters:
        model_name (str): The name of the model.

        Returns:
        str: The path on local drive to the latest checkpoint.
        """
        checkpoint_repo = self.get_checkpoint_local_path(model_name)
        return tf.train.latest_checkpoint(checkpoint_repo)
    
    @override
    def get_checkpoint_local_path(self, model_name: str) -> str:
        checkpoint_repo = os.getenv(VariableNames.CHECKPOINT_REPO_DIR_PATH)
        path = os.path.join(checkpoint_repo, model_name)
        os.makedirs(path, exist_ok=True)
        return path

    @override
    def save_checkpoint(self, model_name: str, checkpoint: tf.train.Checkpoint) -> None:
        directory = self.get_checkpoint_local_path(model_name)
        file_prefix = f"{model_name}_chkpt"
        return checkpoint.save(file_prefix=f"{directory}/{file_prefix}")

    @override
    def save_performance_metrics(self, epoch:int, model_name: str, metrics: dict, descriptions: dict) -> None:
        file_path = os.path.join(self.get_model_repository_local(model_name), "performance.txt")
        file.write(f"Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Timestamp: {dt.datetime.now().timestamp()}\n")
        with open(file_path, 'a') as file:
            for metric, value in metrics.items():
                description_value = descriptions.get(metric, "No description available")
                file.write(f"Epoch {epoch}: {metric} = {value} | Description: {description_value}\n")