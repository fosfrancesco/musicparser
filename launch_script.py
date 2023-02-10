from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

from musicparser.data_loading import TSDataModule
from musicparser.models import ArcPredictionModel

num_workers = 4

datamodule = TSDataModule(batch_size=1, num_workers=num_workers)
datamodule.setup()




