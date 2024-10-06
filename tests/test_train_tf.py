import unittest
import os
import numpy as np
import tensorflow as tf
from keras import preprocessing
from PIL import Image
from brainhealth.models.brain_mri import BrainMri
from brainhealth.metrics.evaluation_metrics import F1Score

class TestAlzheimerDetectionModel(unittest.TestCase):
    def setUp(self):
        self.model_instance = BrainMri()
        self.pretrained_model = self.model_instance.pretrained_model
        self.model = self.model_instance.define_model(self.pretrained_model)

    def test_predict_with_valid_input(self):
        # Load a sample image and preprocess it
        image_path = os.path.expanduser('tests/data/demented/OAS1_0003_MR1_mpr_n4_anon_111_t88_gfc.img_ASL_40.jpg')
        image = Image.open(image_path)
        input_data = np.array(image)
        input_data.resize(32, 32)
        reshaped_input = np.stack([input_data] * 3, axis=-1)  # Stack along the last axis to create 3 channels
        reshaped_input = np.expand_dims(reshaped_input, axis=0)  # Add batch dimension
        test_data_dir = os.path.expanduser('tests/data')

        # Fit model to test dataset
        dataset = preprocessing.image_dataset_from_directory(
            test_data_dir,
            image_size=(32, 32),
            color_mode='rgb',
            batch_size=32,
            seed=123
        )
        self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossEntropy(),
                metrics=[F1Score()]
            )
        self.model.fit(dataset)

        # Make prediction
        predictions = self.model_instance.predict(self.model, reshaped_input)
        
        # Check if the prediction output is as expected
        self.assertEqual(predictions.shape, (1, 1))  # Assuming binary classification
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating))

    def test_predict_with_invalid_input(self):
        # Create an invalid input (e.g., wrong shape)
        invalid_input = np.random.rand(10, 10, 3)  # Invalid shape

        with self.assertRaises(ValueError):
            self.model_instance.predict(self.model, invalid_input)

    def test_skewness(self):
        # test to check if the predictions are skewed
        # return results with warning for skewed predictions
        pass

if __name__ == '__main__':
    unittest.main()