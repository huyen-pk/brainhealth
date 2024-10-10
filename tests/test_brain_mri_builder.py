import pytest
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from brainhealth.models.builders.brain_mri_builder import BrainMriModelBuilder
from brainhealth.metrics.evaluation_metrics import F1Score
from brainhealth.models.brain_result import BrainResult
from brainhealth.models.params import ModelParams, TrainingParams
from brainhealth.models.enums import ModelType, ModelOptimizers

class TestBrainMriModelBuilder():

    @pytest.fixture(scope="function")
    def fixup(self):
        model_params = ModelParams(
            model_name='Test_Brain_MRI',
            base_model_path='../base_models/DBN_model.h5',
            base_model_type=ModelType.Keras,
            models_repo_path='data/models',
            model_dir='data/models/Test_Brain_MRI')
        
        training_params = TrainingParams(
            dataset_path='data/alzheimer',
            batch_size=32,
            num_epoch=10,
            learning_rate=0.001,
            optimizer=ModelOptimizers.Adam,
            kfold=5
        )

        model_builder = BrainMriModelBuilder(model_params=model_params, training_params=training_params)
        pretrained_model = model_builder.base_model
        model = model_builder.define_model(pretrained_model, training_params=training_params)
        yield {"model_builder": model_builder, "model": model}

    def test_model_should_have_correct_architecture(self, fixup):
        model = fixup["model"]
        assert len(model.layers) == 10
    
    def test_model_should_be_tensorflow_format(self, fixup):
        model = fixup["model"]
        assert type(model) is tf.keras.Model    

    def test_model_should_have_correct_input_shape(self, fixup):
        model = fixup["model"]
        assert model.input_shape == (None, 32, 32, 3)

    def test_model_should_have_correct_output_shape(self, fixup):
        model = fixup["model"]
        assert model.output_shape == (None, 1)

    def test_model_should_have_correct_loss_function(self, fixup):
        model = fixup["model"]
        assert model.loss is tf.keras.losses.CategoricalCrossentropy()
    
    def test_model_should_have_correct_optimizer(self, fixup):
        model = fixup["model"]
        assert model.optimizer == 'Adam'

    def test_predict_with_valid_input_should_return_valid_result(self, fixup, capfd):
        valid_input = np.random.rand(1, 32, 32, 3)
        result = fixup["model_builder"].predict(fixup["model"], valid_input)
        print("output predictions", result.predictions)

        captured = capfd.readouterr()
        print(captured.out)

        assert type(result) is BrainResult

    @pytest.mark.parametrize("batch_size", [2, 3, 4])
    def test_predict_with_batch_of_images_should_return_prediction_per_image(self, fixup, batch_size):
        valid_input = np.random.rand(batch_size, 32, 32, 3)
        result = fixup["model_builder"].predict(fixup["model"], valid_input)

        assert(len(result.predictions) == batch_size)
        assert type(result) is BrainResult

    @pytest.mark.parametrize("batch_size, width, height, color_channels", 
                            [
                                (1, 10, 10, 3), # wrong size
                                (1, 20, 32, 3), # wrong width
                                (1, 32, 20, 3), # wrong height
                                (1, 32, 32, 2), # wrong color channels
                                (1, 32, 32, 1), # wrong color channels
                                (1, 32, 32, 4), # wrong color channels
                            ],)
    def test_predict_with_invalid_input_should_raise_exception(self, fixup, batch_size, width, height, color_channels):
        invalid_input = np.random.rand(batch_size, width, height, color_channels)
        with pytest.raises(ValueError):
            fixup["model_builder"].predict(fixup["model"], invalid_input)