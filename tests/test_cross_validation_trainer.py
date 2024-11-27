import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pytest
import tensorflow as tf
from keras import layers
from keras import models
import numpy as np
from brainhealth.metrics.evaluation_metrics import F1Score
from brainhealth.models.brain_result import BrainResult
from brainhealth.models.params import ModelParams, TrainingParams
from brainhealth.models.enums import ModelType, ModelOptimizers
class TestBrainMriModelBuilder():
    def fixup(self):
        model_params = ModelParams(
            model_name='Test_Brain_MRI',
            base_model_name='../base_models/DBN_model.h5',
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
        model = tf.keras.Sequential()
        yield {"model": model, "model_params": model_params, "training_params": training_params}

    def test_model_performance_should_be_evaluated_by_parameterized_metric(self, fixup):
        pass

    def test_training_should_be_kfold(self, fixup):
        pass

