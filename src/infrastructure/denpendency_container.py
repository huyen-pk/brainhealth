import os
from injector import Injector, Module, threadlocal, provider
from brainhealth.utilities.storage import Storage,S3Storage
from brainhealth.models.conf import VariableNames

class AppModule(Module):
    @provider
    @threadlocal
    def provide_storage(self) -> Storage:
        # Here you can configure the Database instance if needed
        storage = S3Storage(os.getenv(VariableNames.BUCKET_NAME))
        return storage
    

def configure_injector() -> Injector:
    injector = Injector([AppModule()])
    return injector
