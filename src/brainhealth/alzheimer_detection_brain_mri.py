import os
from brainhealth.models.builders.brain_mri_builder import BrainMriModelBuilder
from brainhealth.models.trainers.cross_validation_trainer import CrossValidationTrainer_TF as Trainer
from brainhealth.models.params import ModelParams, TrainingParams
from brainhealth.models.enums import ModelType, ModelOptimizers
from brainhealth.metrics.evaluation_metrics import F1Score
from brainhealth.models.conf import VariableNames
import numpy as np

class AlzheimerDetectionBrainMri:
    def __init__(self) -> None:
        training_params = TrainingParams(
            dataset_path=os.getenv(VariableNames.TRAIN_DATA_DIR),
            batch_size=32,
            num_epoch=10,
            learning_rate=0.001,
            optimizer=ModelOptimizers.Adam,
            kfold=5
        )
        model_params = ModelParams(
            model_name='AlzheimerDetectionBrainMRI',
            base_model_path=os.getenv(VariableNames.BASE_MODEL_PATH),
            base_model_type=ModelType.Keras,
            models_repo_path=os.getenv(VariableNames.MODELS_REPO_DIR_PATH))
        model_params.models_repo_path=os.getenv(VariableNames.MODELS_REPO_DIR_PATH)
        model_params.model_dir=os.path.join(model_params.model_name, model_params.models_repo_path)
        
        # Define the model
        builder = BrainMriModelBuilder()
        base_model = builder.load_base_model(model_params.base_model_type, model_params.base_model_path)
        model = builder.define_model(base_model, training_params)
        
        # Train & evaluate the model
        trainer = Trainer()
        self.tuned_model, self.tuned_model_path = trainer.train(model, model_params, training_params, F1Score.__name__)

    def predict(self, dir) -> tuple[np.ndarray, np.ndarray]:
        trainer = Trainer()
        dataset = trainer.__load_data__(dir, shuffle=False)
        file_names = dataset.list_files(shuffle=False)
        predictions = self.tuned_model.predict(dataset)
        return (file_names, predictions)