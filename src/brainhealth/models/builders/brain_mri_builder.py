import tensorflow as tf
from keras import layers, models
from brainhealth.models import params
from brainhealth.models.builders.builder_base import ModelBuilderBase
from infrastructure.units_of_work import ModelTrainingDataDomain
import numpy as np

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

        # Define the augmentation layers
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip horizontally and vertically
            layers.RandomRotation(0.2),                    # Randomly rotate by 20%
            layers.RandomZoom(0.2),                        # Random zoom by 20%
            layers.RandomContrast(0.2),                    # Random contrast adjustment
            layers.RandomBrightness(0.2)                   # Random brightness adjustment
        ])
        input_shape = (32, 32, 3)
        # Define the model
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_shape)
        ])
        for layer in data_augmentation.layers:
            model.add(layer)

        if base_model is not None:
            for layer in base_model.layers:
                model.add(layer)
        
        model.summary()
        return model
    
    def fetch_data(self, 
                   page_index: int, 
                   training_params: params.TrainingParams,
                   **kwargs) -> tuple[np.ndarray, np.ndarray]:
        images, labels = self.data_domain.get_dataset(
                    page_size=training_params.batch_size,
                    page_index=page_index,
                    page_count=1,
                    continuation_token=kwargs.get('continuation_token', None))
        return images, labels