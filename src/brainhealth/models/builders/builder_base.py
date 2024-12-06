import os
import copy
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from keras import models
from keras import Model as Model, optimizers as ops, Optimizer
from brainhealth.models import params
from infrastructure.units_of_work import ModelTrainingDataDomain
from concurrent.futures import ThreadPoolExecutor
import uuid

class ModelBuilderBase(ABC):

    def __init__(self, data_domain: ModelTrainingDataDomain) -> None:
        self.data_domain = data_domain
        self.model = None
        self.checkpoint = None
        self.model_params = None
        self.training_params = None

    @abstractmethod
    def fetch_data(self, 
                   page_index: int, 
                   training_params: params.TrainingParams,
                   model_params: params.ModelParams,
                   **kwargs) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def define_model(self, 
                     base_model: models.Model,
                     model_params: params.ModelParams) -> tf.keras.Model:
        """
        Define a model for training.

        Parameters:
        base_model (models.Model): The base model to use for training.
        model_params (params.ModelParams): The parameters to use for training.

        Returns:
        tf.keras.Model: The defined model.
        """
        pass

    def load_base_model(self, model_name: str) -> Model:
        """
        Load a pre-trained model from a file path.

        Parameters:
        model_name (str): The name of the model to load.

        Returns:
        tf.keras.Model: The pre-trained model.
        """
        return self.data_domain.get_model(model_name=model_name, file_type="h5")


    def save_checkpoint(self, model_name:str, checkpoint: tf.train.Checkpoint) -> None:
        self.data_domain.save_checkpoint(model_name=model_name, checkpoint=checkpoint)
    
    def load_checkpoint(self, 
                        model_name: str,
                        model: Model, 
                        optimizer: ops.Optimizer) -> tf.train.Checkpoint:
        """
        Load a model from a checkpoint directory.

        Parameters:
        model_name (str): name of the model to load checkpoints for.
        model: The model to load the checkpoint into.
        optimizer: The optimizer to load the checkpoint into.

        Returns:
        tf.train.Checkpoint: checkpoint.
        """
        latest_checkpoint = self.data_domain.get_latest_checkpoint(model_name=model_name)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"Restored from {latest_checkpoint}")
        else:
            print("No checkpoint found, initializing from scratch.")
        return checkpoint
    
    def load_weights(self, 
                     model: Model, 
                     model_name: str) -> Model:
        """
        Load the weights of a model from a file path.

        Parameters:
        model (tf.keras.Model): The model to load the weights into.
        model_name (str): The name of the model to load the weights from.

        Returns:
        tf.keras.Model: The model with the loaded weights.
        """
        weights_path = self.data_domain.get_weights(model_name)
        if weights_path:
            model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
        else:
            print("No weights found, initializing from scratch.")
        return model
    @abstractmethod
    def init_model(self, model: models.Model, 
                   model_params: params.ModelParams,
                   training_params: params.TrainingParams) -> tuple[models.Model, tf.train.Checkpoint, Optimizer]:
        """
        Initialize the model with the training parameters.

        Parameters:
        model (tf.keras.Model): The model to initialize.
        model_params (params.ModelParams): The model parameters to use for initialization.
        training_params (params.TrainingParams): The training parameters to use for initialization.

        Returns:
        tf.keras.Model: The initialized model.
        tf.train.Checkpoint: The model checkpoint.
        """
        pass
    
    def build(self, 
              model_params: params.ModelParams,
              training_params: params.TrainingParams
              ) -> tuple[models.Model, tf.train.Checkpoint]:
        # Load the base model
        base_model = self.load_base_model(
            model_name=model_params.base_model_name
        )
        # Define the model
        model = self.define_model(base_model=base_model, model_params=model_params)
        # Initialize the model
        model = self.init_model(model=model, model_params=model_params, training_params=training_params)
        self.model = model
        self.model_params = model_params
        self.training_params = training_params
        return model
    
    def apply_gradients(self,
                        model: Model, 
                        optimizer: ops.Optimizer, 
                        input: tf.Tensor, 
                        labels: tf.Tensor) -> tuple[ops.Optimizer, tf.Tensor, dict]:
        """
        Perform a single training step.

        Parameters:
        model (Model): The model to train.
        optimizer (ops.Optimizer): The optimizer to use for training.
        data (tf.Tensor): The input data.
        labels (tf.Tensor): The target labels.

        Returns:
        Optimizer: The updated optimizer.
        tf.Tensor: The loss value.
        dict: The computed metrics.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs=input, training=True)
            loss = model.compute_loss(x=input, y=labels, y_pred=predictions)
            metrics = model.compute_metrics(x=input, y=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return optimizer, loss, metrics

    def train(self, 
              model: Model, 
              model_params: params.ModelParams, 
              training_params: params.TrainingParams,
              checkpoint: tf.train.Checkpoint) -> tuple[Model, str]:
        
        if model is None:
            raise ValueError('Model is required for training')
        
        if model_params is None:
            raise ValueError('Model parameters are required for training')
        
        if training_params is None:
            raise ValueError('Training parameters are required for training')

        tuned_model = copy.deepcopy(model)

        optimizer = ops.get(tuned_model.optimizer)  
        save_every_n_batches = training_params.save_every_n_batches
        # Prepare the data
        steps_per_epoch = training_params.steps_per_epoch # Set according to the streaming data availability
        continuation_token  = None
        cycle_identifier = str(uuid.uuid4())
        for epoch in range(1, training_params.num_epoch + 1):
            print(f"Epoch {epoch}/{training_params.num_epoch}")
            if(epoch == 1):
                # Freeze all layers except fully connected layers to train for the first epoch
                for layer in tuned_model.layers:
                    if 'dense' not in layer.name:
                        layer.trainable = False
            else:
                # Unfreeze all layers and train the model until convergence
                for layer in tuned_model.layers:
                    layer.trainable = True

            for step in range(1, steps_per_epoch  + 1):
                # Fetch a batch of data from the stream
                batches, labels = self.fetch_data(
                    page_index=step, 
                    training_params=training_params, 
                    model_params=model_params,
                    continuation_token=continuation_token
                )
                batchX = batches[0]
                batchX_labels = labels[0]
                # Validate the model on the last step of an epoch
                if step % steps_per_epoch == 0:
                    results = tuned_model.evaluate(x=batchX, y=batchX_labels, steps=1, return_dict=True, verbose=0)
                    descriptions = {}
                    for metric, value in results.items():
                        descriptions[f"{metric}"] = value
                        print(f"{metric}: {value}")
                    self.data_domain.save_performance_metrics(
                        epoch=epoch, model_name=model_params.model_name, metrics=results, descriptions=descriptions, identifier=f"{cycle_identifier}_validation")
                else:
                    # Perform a single training step
                    optimizer, loss, metrics = self.apply_gradients(model=tuned_model, optimizer=optimizer, input=batchX, labels=batchX_labels)
                    self.data_domain.save_performance_metrics(epoch=epoch, model_name=model_params.model_name, metrics=metrics, descriptions={"step": step}, identifier=f"{cycle_identifier}_training")

                if step % save_every_n_batches == 0:
                    # Send command to save checkpoint to storage
                    self.data_domain.save_checkpoint(model_name=model_params.model_name, checkpoint=checkpoint)
                    print(f"Checkpoint saved at batch {step} in epoch {epoch} in {self.data_domain.get_checkpoint_local_path(model_params.model_name)}")
                
                self.data_domain.purge_dataset(model_name=model_params.model_name, batch_index=step)

        # Save the model
        final_model_path = self.data_domain.save_model(model=tuned_model, model_name=model_params.model_name, type='keras')
        return (tuned_model, final_model_path)


    # def evaluate(model: Model) -> dict:
    #     """
    #     Evaluate a model on a dataset.

    #     Parameters:
    #     model (tf.keras.Model): The model to evaluate.
    #     data (np.ndarray): The input data.
    #     labels (np.ndarray): The target labels.

    #     Returns:
    #     dict: The computed metrics.
    #     """

    #     results = model.evaluate(x=data, y=labels, return_dict=True)
    #     return results