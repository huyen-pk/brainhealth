import os
from injector import Injector, Module, threadlocal, singleton, provider
from infrastructure.storage import S3Storage, LocalStorage, BlobStorage
from brainhealth.models.builders.brain_mri_builder import BrainMriModelBuilder
from brainhealth.models.conf import VariableNames
from infrastructure.units_of_work import ModelTrainingDataDomain
from infrastructure.repositories import ModelRepository, S3ImageDatasetRepository, CheckpointRepository
class AppModule(Module):
   
    @provider
    @singleton
    def provide_model_repository(self) -> ModelRepository:
        return ModelRepository(storage=S3Storage(os.getenv(VariableNames.MODEL_BUCKET_NAME), None, None))
    
    @provider
    @singleton
    def provide_checkpoint_repository(self) -> CheckpointRepository:
        return CheckpointRepository(storage=S3Storage(os.getenv(VariableNames.CHECKPOINT_BUCKET_NAME), None, None))

    @provider
    @singleton
    def provide_dataset_repository(self) -> S3ImageDatasetRepository:
        return S3ImageDatasetRepository(storage=S3Storage(os.getenv(VariableNames.DATASET_BUCKET_NAME), None, None))

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
   
class DependencyContainer:
    def configure_injector() -> Injector:
        injector = Injector([AppModule()])
        return injector
