import tensorflow as tf
from keras import layers, models, optimizers, Optimizer, metrics, losses
from keras import initializers
from brainhealth.models import enums, params
from brainhealth.models.builders.builder_base import ModelBuilderBase
from brainhealth.metrics.evaluation_metrics import F1Score
from infrastructure.units_of_work import ModelTrainingDataDomain
import numpy as np
from typing import override

class BrainMriModelBuilder(ModelBuilderBase):

    def __init__(self, data_domain: ModelTrainingDataDomain) -> None:
        self.data_domain = data_domain

    def define_model(self, 
                     base_model: models.Model,
                     model_params: params.ModelParams) -> tf.keras.Model:
        """
        Define the model architecture based on the foundation model and the training parameters.

        Parameters:
        base_model (tf.keras.Model): The foundation model to build upon.
        training_params (params.TrainingParams): The training parameters to use for the model.

        Returns:
        tf.keras.Model: The compiled model.
        """

        # TODO Define image processing layers

        input_shape = (32, 32, 3)
        # Define the model
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_shape)
        ])

        # Define the augmentation layers
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip horizontally and vertically
            layers.RandomRotation(0.2),                    # Randomly rotate by 20%
            layers.RandomZoom(0.2),                        # Random zoom by 20%
            layers.RandomContrast(0.2),                    # Random contrast adjustment
            layers.RandomBrightness(0.2)                   # Random brightness adjustment
        ])

        dense = layers.Dense(1024, 
                activation='relu', 
                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))
        
        dropout = layers.Dropout(rate=0.85)

        dense_last = layers.Dense(1, activation='sigmoid')

        for layer in data_augmentation.layers:
            model.add(layer)

        if base_model is not None:
            for layer in base_model.layers[:-2]:
                model.add(layer)
        model.add(dense)
        model.add(dropout)
        model.add(dense_last)
        model.summary()
        return model
    
    @override
    def init_model(self, model: models.Model, 
                   model_params: params.ModelParams,
                   training_params: params.TrainingParams) -> tuple[models.Model, tf.train.Checkpoint, Optimizer]:
        """
        Initialize the model with the training parameters.

        Parameters:
        model (tf.keras.Model): The model to initialize.
        model_params (params.ModelParams): The model parameters to use for initialization.

        Returns:
        tf.keras.Model: The initialized model.
        tf.train.Checkpoint: The model checkpoint.
        """
        # Load weights if available

        # Define the optimizer
        optimizer=None

        if training_params.optimizer == enums.ModelOptimizers.Adam:
            optimizer = optimizers.Adam(learning_rate=training_params.learning_rate)
        elif training_params.optimizer == enums.ModelOptimizers.SGD:
            optimizer = optimizers.SGD(learning_rate=training_params.learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {training_params.optimizer}')

        # Load checkpoints if available
        checkpoint = self.load_checkpoint(
            model_name=model_params.model_name,
            model=model, 
            optimizer=optimizer)
        
        # Initialize the model's variables and inputs to avoid unknown variable error
        input_shape = model.input_shape # Get the shape of the input layer, which will be (None, shape)
        dummy_input = tf.zeros((1, *input_shape[1:]))
        __ = model(dummy_input)
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=losses.BinaryCrossentropy(),
            metrics=[metrics.Precision(), metrics.Recall(), F1Score()]
        )
        return model, checkpoint, optimizer

    def fetch_data(self, 
                   page_index: int, 
                   training_params: params.TrainingParams,
                   model_params: params.ModelParams,
                   **kwargs) -> tuple[np.ndarray, np.ndarray]:
        images, labels = self.data_domain.get_dataset(
                    model_name=model_params.model_name,
                    page_size=training_params.batch_size,
                    page_index=page_index,
                    page_count=1,
                    continuation_token=kwargs.get('continuation_token', None))
        return images, labels