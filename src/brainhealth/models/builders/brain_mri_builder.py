import os
import tensorflow as tf
from keras import layers, models
from keras import utils
from brainhealth.models import enums, params
import tempfile

class BrainMriModelBuilder:
    def load_base_model(self, 
                        model_type: enums.ModelType, 
                        model_file_path: str) -> tf.keras.Model:
        """
        Load a pre-trained model from a file path.

        Parameters:
        model_type (enums.ModelType): The type of the model to load.
        model_path (str): The file path to the pre-trained model.

        Returns:
        tf.keras.Model: The pre-trained model.
        """
        if model_file_path is None:
            raise ValueError('Model file path is required.')
        
        # Attempt to download the model if the file does not exist
        if not os.path.exists(model_file_path):
            temp_file = tempfile.NamedTemporaryFile(delete=False).name
            model_file_path = utils.get_file(temp_file, model_file_path)
            if not os.path.exists(model_file_path) or os.path.getsize(model_file_path) == 0:
                raise FileNotFoundError(f'Model not found at {model_file_path}')
        
        if model_type == enums.ModelType.Keras:
            return models.load_model(model_file_path, compile=False)
        elif model_type == enums.ModelType.PyTorch:
            raise NotImplementedError('PyTorch model conversion to TensorFlow is not supported yet.')
        else:
            raise ValueError(f'Unsupported model type: {model_type}')

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
        
        # Define the model
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(32, 32, 3))
        ])
        for layer in data_augmentation.layers:
            model.add(layer)

        if base_model is not None:
            for layer in base_model.layers:
                model.add(layer)

        model.summary()
        return model
    
    def save_model(self, model: models.Model, model_dir: str) -> str:
        """
        Save the model to a directory.

        Parameters:
        model (tf.keras.Model): The model to save.
        model_dir (str): The directory to save the model to.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(model_dir)