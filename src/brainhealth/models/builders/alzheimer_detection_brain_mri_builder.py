import tensorflow as tf
from keras import layers, models, optimizers, Optimizer, metrics, losses, Model
from keras import initializers
from brainhealth.models import enums, params
from brainhealth.models.builders.builder_base import ModelBuilderBase
from brainhealth.metrics.evaluation_metrics import F1Score
from infrastructure.units_of_work import ModelTrainingDataDomain
import numpy as np
from typing import override

class AlzheimerDetectionBrainMriModelBuilder(ModelBuilderBase):

    def __init__(self, data_domain: ModelTrainingDataDomain) -> None:
        self.data_domain = data_domain

    @override
    def load_base_model(self, model_name: str) -> Model:
        """
        Load a pre-trained model from a file path.

        Parameters:
        model_name (str): The name of the model to load.

        Returns:
        tf.keras.Model: The pre-trained model.
        """
        return self.data_domain.get_model(model_name=model_name, file_type="keras")
    
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

        return base_model
    
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
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=training_params.learning_rate,
            decay_steps=training_params.steps_per_epoch,
            decay_rate=0.95,
            staircase=True  # If True, learning rate decays in discrete steps
        )

        # Define the optimizer
        optimizer=None

        if training_params.optimizer == enums.ModelOptimizers.Adam:
            optimizer = optimizers.Adam(learning_rate=lr_schedule)
        elif training_params.optimizer == enums.ModelOptimizers.SGD:
            optimizer = optimizers.SGD(learning_rate=lr_schedule)
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