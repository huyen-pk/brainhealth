import os
import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
from keras import metrics
from brainhealth.metrics.evaluation_metrics import F1Score
from brainhealth.models import params
from brainhealth.models import enums

class BrainMriModelBuilder:
    def __init__(self, model_params: params.ModelParams, training_params: params.TrainingParams):
        self.model_params = model_params
        self.training_params = training_params
        self.models_repo = model_params.models_repo_path
        self.base_model_path = model_params.base_model_path
        self.base_model = models.load_model(self.base_model_path, compile=False)
        self.model_name = 'DeepBrainNet_Alzheimer'
        self.model_dir = os.path.join(self.models_repo, self.model_name)
        self.train_data_dir = training_params.dataset_path
    
    def load_pretrained_model(self, 
                              model_type: enums.ModelType, 
                              model_path: str) -> tf.keras.Model:
        """
        Load a pre-trained model from a file path.

        Parameters:
        model_type (enums.ModelType): The type of the model to load.
        model_path (str): The file path to the pre-trained model.

        Returns:
        tf.keras.Model: The pre-trained model.
        """
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found at {model_path}')
        
        if model_type == enums.ModelType.Keras:
            return models.load_model(model_path, compile=False)
        elif model_type == enums.ModelType.PyTorch:
            raise NotImplementedError('PyTorch model conversion to TensorFlow is not supported yet.')
        else:
            raise ValueError(f'Unsupported model type: {model_type}')

    def define_model(self, 
                     base_model: tf.keras.Model, 
                     training_params: params.TrainingParams) -> tf.keras.Model:
        """
        Define the model architecture based on the foundation model and the training parameters.

        Parameters:
        base_model (tf.keras.Model): The foundation model to build upon.
        training_params (params.TrainingParams): The training parameters to use for the model.

        Returns:
        tf.keras.Model: The compiled model.
        """
        # Define the augmentation layers
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip horizontally and vertically
            layers.RandomRotation(0.2),                    # Randomly rotate by 20%
            layers.RandomZoom(0.2),                        # Random zoom by 20%
            layers.RandomContrast(0.2),                    # Random contrast adjustment
            layers.RandomBrightness(0.2)                   # Random brightness adjustment
        ])
        
        # Define the model
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(32, 32, 3))
        ])
        for layer in data_augmentation.layers:
            model.add(layer)

        if base_model is not None:
            for layer in base_model.layers:
                model.add(layer)

        optimizer=None

        if self.training_params.optimizer == enums.ModelOptimizers.Adam:
            optimizer = optimizers.Adam(learning_rate=training_params.learning_rate)
        elif self.training_params.optimizer == enums.ModelOptimizers.SGD:
            optimizer = optimizers.SGD(learning_rate=training_params.learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {training_params.optimizer}')

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[metrics.Precision(), metrics.Recall(), F1Score()]
        )
        model.summary()
        return model