import os
from injector import Injector, Module, threadlocal, singleton, provider
from infrastructure.storage import Storage, S3Storage, LocalStorage
from brainhealth.models.conf import VariableNames
from brainhealth.models.trainers import trainer_base, cross_validation_trainer as cvt, trainer_with_streaming_data as tsd

class AppModule(Module):
    @provider
    @threadlocal
    def provide_storage(self) -> Storage:
        storage = S3Storage(os.getenv(VariableNames.BUCKET_NAME))
        return storage
    
    @provider
    @threadlocal
    def provide_trainer(self, storage: Storage) -> trainer_base.Trainer:
        return tsd.StreamingData_TF_Trainer(storage)

class AppModuleLocal(Module):
    @provider
    @singleton
    def provide_storage(self) -> Storage:
        return LocalStorage(os.getenv(VariableNames.TRAIN_DATA_DIR))
    
    @provider
    @threadlocal
    def provide_trainer(self) -> trainer_base.Trainer:
        return cvt.CrossValidationTrainer_TF()
    
class DependencyContainer:
    def configure_injector() -> Injector:
        injector = Injector([AppModule()])
        return injector
    
    def configure_injector_local() -> Injector:
        injector = Injector([AppModuleLocal()])
        return injector