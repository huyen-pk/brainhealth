import os
from brainhealth.models.conf import VariableNames
from brainhealth.models.builders.alzheimer_detection_brain_mri_builder import AlzheimerDetectionBrainMriModelBuilder
from brainhealth.models.params import ModelParams, TrainingParams
from brainhealth.models.enums import ModelOptimizers, ModelType
from infrastructure.denpendency_container import DependencyContainer

print("model: ", os.getenv(VariableNames.MODELS_REPO_DIR_PATH))
print("checkpoint:", os.getenv(VariableNames.CHECKPOINT_REPO_DIR_PATH))
print("dataset: ", os.getenv(VariableNames.TRAIN_DATA_DIR))

di_container = DependencyContainer.configure_injector_local()
builder = di_container.get(AlzheimerDetectionBrainMriModelBuilder)

training_params = TrainingParams(
            dataset_path=None,
            batch_size=32,
            num_epoch=10,
            steps_per_epoch=100,
            learning_rate=0.00005,
            optimizer=ModelOptimizers.Adam,
            save_every_n_batches=100
        )
model_params = ModelParams(
            model_name='AlzheimerDetectionBrainMRI',
            base_model_name='AlzheimerDetectionBrainMRI',
            base_model_type=ModelType.Keras,
            models_repo_path=None)
        
       
model, checkpoint, optimizer = builder.build(
            model_params=model_params,
            training_params=training_params
        )
# Train & evaluate the model
tuned_model, tuned_model_path = builder.train(
            model=model, model_params=model_params, training_params=training_params, checkpoint=checkpoint)
