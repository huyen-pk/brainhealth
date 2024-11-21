import os
from abc import ABC, abstractmethod
from infrastructure.repositories import ModelRepository, CheckpointRepository, S3ImageDatasetRepository
import tensorflow as tf
from keras import Model
import numpy as np

class ModelTrainingDataDomain(ABC):
    def __init__(self, 
                 model_repository: ModelRepository, 
                 checkpoint_repository: CheckpointRepository,
                 dataset_repository: S3ImageDatasetRepository) -> None:
        self.model_repository = model_repository
        self.checkpoint_repository = checkpoint_repository
        self.dataset_repository = dataset_repository

    def get_model(self, model_name: str) -> Model:
        return self.model_repository.get(model_name)

    def save_model(self, file_path, model):
        self.model_repository.save(file_path, model)

    def get_model_repository_local(self, model_name: str) -> str:
        return self.model_repository.get_local_path(model_name)
    
    def get_dataset(self, page_size, page_index, page_count=1, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self.dataset_repository.get(page_size, page_index, page_count, **kwargs)
    
    def purge_dataset(self, model_name: str, batch_index: int) -> None:
        local_path = self.dataset_repository.get_local_path(model_name, batch_index)
        if os.path.exists(local_path):
            os.rmdir(local_path)

    def get_latest_checkpoint(self, model_name) -> str:
        """
        Get the latest checkpoint for a model.

        Parameters:
        model_name (str): The name of the model.

        Returns:
        str: The path on local drive to the latest checkpoint.
        """
        return self.checkpoint_repository.get_latest(model_name)

    def save_checkpoint(self, model_name: str, checkpoint: tf.train.Checkpoint) -> None:
        self.checkpoint_repository.save_upload(model_name=model_name, checkpoint=checkpoint)

    def save_performance_metrics(self, epoch:int, model_name: str, metrics: dict) -> None:
        self.model_repository.save_performance_metrics(epoch=epoch, model_name=model_name, metrics=metrics)