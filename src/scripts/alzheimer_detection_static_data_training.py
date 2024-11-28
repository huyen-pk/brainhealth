import os
from brainhealth.models.conf import VariableNames
import argparse
from brainhealth.models.builders.brain_mri_builder import BrainMriModelBuilder
from brainhealth.models.params import ModelParams, TrainingParams
from brainhealth.models.enums import ModelOptimizers, ModelType
from infrastructure.denpendency_container import DependencyContainer

parser = argparse.ArgumentParser(description='Alzheimer Detection on Brain MRI')
parser.add_argument('--data', type=str, help='Directory containing MRI images for training')
parser.add_argument('--model', type=str, help='Path to foundation model to build our model upon')
parser.add_argument('--checkpoint', type=str, help='Model repository')
args = parser.parse_args()

print("model: ", os.getenv(VariableNames.MODEL_STORAGE_CONNECTION_STRING))
print("checkpoint:", os.getenv(VariableNames.CHECKPOINT_STORAGE_CONNECTION_STRING))
print("dataset: ", os.getenv(VariableNames.DATASET_STORAGE_CONNECTION_STRING))

model_storage = args.model if args.model else os.getenv(VariableNames.MODEL_STORAGE_CONNECTION_STRING)
checkpoint_storage = args.checkpoint if args.checkpoint else os.getenv(VariableNames.CHECKPOINT_STORAGE_CONNECTION_STRING)
dataset_storage = args.data if args.data else os.getenv(VariableNames.DATASET_STORAGE_CONNECTION_STRING)

di_container = DependencyContainer.configure_injector()
builder = di_container.get(BrainMriModelBuilder)

training_params = TrainingParams(
            dataset_path=None, # None if pulling data from cloud storage
            batch_size=32,
            num_epoch=10,
            learning_rate=0.001,
            optimizer=ModelOptimizers.Adam,
            kfold=5
        )
model_params = ModelParams(
            model_name='AlzheimerDetectionBrainMRI',
            base_model_name='DeepBrainNet',
            base_model_type=ModelType.Keras,
            models_repo_path=None)
        
       
model, checkpoint, optimizer = builder.build(
            model_params=model_params,
            training_params=training_params
        )
# Train & evaluate the model
tuned_model, tuned_model_path = builder.train(
            model=model, model_params=model_params, training_params=training_params, checkpoint=checkpoint)
