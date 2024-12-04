import os
from injector import Injector, Module, threadlocal, singleton, provider
from infrastructure.storage import S3Storage, LocalStorage, BlobStorage
from brainhealth.models.builders.brain_mri_builder import BrainMriModelBuilder
from brainhealth.models.conf import VariableNames
from infrastructure.units_of_work import ModelTrainingDataDomain, Local_ModelTrainingDataDomain
from infrastructure.repositories import ModelRepository, S3ImageDatasetRepository, CheckpointRepository
class AppModule(Module):
   
    @provider
    @singleton
    def provide_model_repository(self) -> ModelRepository:
        return ModelRepository(storage=S3Storage(os.getenv(VariableNames.MODEL_STORAGE_CONNECTION_STRING)))
    
    @provider
    @singleton
    def provide_checkpoint_repository(self) -> CheckpointRepository:
        return CheckpointRepository(storage=S3Storage(os.getenv(VariableNames.CHECKPOINT_STORAGE_CONNECTION_STRING)))

    @provider
    @singleton
    def provide_dataset_repository(self) -> S3ImageDatasetRepository:
        return S3ImageDatasetRepository(storage=S3Storage(os.getenv(VariableNames.DATASET_STORAGE_CONNECTION_STRING)))

    @provider
    @threadlocal
    def provide_ModelTrainingDataDomain(self,
                                        model_repository: ModelRepository,
                                        checkpoint_repository: CheckpointRepository,
                                        dataset_repository: S3ImageDatasetRepository ) -> ModelTrainingDataDomain:
        return ModelTrainingDataDomain(
            model_repository=model_repository, 
            checkpoint_repository=checkpoint_repository, 
            dataset_repository=dataset_repository)

    @provider
    @threadlocal
    def provide_builder(self, dataStorage: ModelTrainingDataDomain) -> BrainMriModelBuilder:
        return BrainMriModelBuilder(dataStorage)
   
class AppModule_Local(Module):
   
    @provider
    @singleton
    def provide_model_repository(self) -> ModelRepository:
        return ModelRepository(storage=LocalStorage(os.getenv(VariableNames.MODELS_REPO_DIR_PATH)))
    
    @provider
    @singleton
    def provide_checkpoint_repository(self) -> CheckpointRepository:
        return CheckpointRepository(storage=LocalStorage(os.getenv(VariableNames.CHECKPOINT_REPO_DIR_PATH)))

    @provider
    @singleton
    def provide_dataset_repository(self) -> S3ImageDatasetRepository:
        return S3ImageDatasetRepository(LocalStorage=S3Storage(os.getenv(VariableNames.TRAIN_DATA_DIR)))

    @provider
    @singleton
    def provide_ModelTrainingDataDomain(self,
                                        model_repository: ModelRepository,
                                        checkpoint_repository: CheckpointRepository,
                                        dataset_repository: S3ImageDatasetRepository ) -> ModelTrainingDataDomain:
        return Local_ModelTrainingDataDomain(
            model_repository=model_repository, 
            checkpoint_repository=checkpoint_repository, 
            dataset_repository=dataset_repository)

    @provider
    @threadlocal
    def provide_builder(self, dataStorage: Local_ModelTrainingDataDomain) -> BrainMriModelBuilder:
        return BrainMriModelBuilder(dataStorage)
    
class DependencyContainer:
    def configure_injector() -> Injector:
        injector = Injector([AppModule()])
        return injector
    
    def configure_injector_local() -> Injector:
        injector = Injector([AppModule_Local()])
        return injector
