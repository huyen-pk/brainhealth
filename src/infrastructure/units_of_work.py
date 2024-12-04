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
        for dir in os.listdir(self.dataset_repo):
            dir_files = []
            directory_path = os.path.join(self.dataset_repo, dir)
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    try:
                        dir_files.append(os.path.join(root, file))
                    except Exception as e:
                        print(f"Error with file {file}: {e}")
            dir_files.sort()
            self.all_files.append(dir_files)
        self.start_index = 0


    @override
    def get_model(self, model_name: str, file_type: str) -> Model:
        model_repo = self.get_model_repository_local(model_name)
        model_file_path = os.path.join(model_repo, f"{model_name}.{file_type}")
        if not os.path.exists(model_file_path) or os.path.getsize(model_file_path) == 0:
                raise FileNotFoundError(f'Model not found at {model_file_path}')
        
        return models.load_model(model_file_path, compile=False)

    @override
    def save_model(self, file_path: str, model_name: str):
        pass

    @override
    def save_model(self, model_name: str, model: Model, type: str = 'keras') -> str:
        repo = self.get_model_repository_local(model_name)
        file_path = os.path.join(repo, f"{model_name}_{str(dt.datetime.now().timestamp())}.{type}")
        model.save(file_path)

    @override
    def get_model_repository_local(self, model_name: str) -> str:
        local_repository = os.getenv(VariableNames.MODELS_REPO_DIR_PATH)
        model_path = os.path.join(local_repository, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        return model_path
    
    @override
    def get_label_classes(self, model_name: str) -> list[str]:
        return os.getenv('TRAIN_DATA_LABELS', '').split(',')   
    
    @override
    def get_dataset(self, model_name:str, page_size:int, page_index:int, page_count=1, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        dataset_repo = self.dataset_repo
        if not os.path.exists(dataset_repo) or not os.listdir(dataset_repo):
            raise FileNotFoundError(f'Dataset repository not found or is empty at {dataset_repo}')
        
        # labels = self.get_label_classes(model_name)
        all_files = self.all_files
        self.start_index = page_index * page_size
        end_index = self.start_index + (page_size * page_count)
        paged_files = []
        for index in range(self.start_index, end_index):
            for cat in all_files:
                if index < len(cat):
                    paged_files.append(cat[index])
                else:
                    paged_files.append(cat[index % len(cat)])

        images = [] # a batch of images
        labels = []
        for file_path in paged_files:
            img = pp.image.load_img(
                path=file_path,
                color_mode='rgb',
                target_size=(32, 32)
            )
            images.append(pp.image.img_to_array(img))
            labels.append(os.path.basename(os.path.dirname(file_path)))

        # Convert labels to float values
        unique_labels = tf.constant(sorted(set(labels)))
        values = tf.range(len(unique_labels), dtype=tf.float32)
        table = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(keys=unique_labels, values=values),
                default_value=-1.0  # Value for unknown labels
                )
        label_set = tf.data.Dataset.from_tensor_slices(labels)
        mapped_labels = label_set.map(lambda label: table.lookup(label))
        encoded_labels = []
        for mapped_label in mapped_labels:
            encoded_labels.append(mapped_label.numpy())

        batches = []
        batches_labels = []
        batches.append(images)
        batches_labels.append(encoded_labels)
        return np.array(batches), np.array(batches_labels)
    
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
        file_path = os.path.join(self.get_model_repository_local(model_name), f"performance{dt.datetime.now().date()}.txt")
        with open(file_path, 'a') as file:
            file.write(f"Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Timestamp: {dt.datetime.now().timestamp()}\n")
            for metric, value in metrics.items():
                file.write(f"Epoch {epoch}: {metric} = {value} | Description: {descriptions[metric]}\n")