from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
import torch
import random
import argparse
import warnings
# warnings.filterwarnings('ignore') # avoid printing the partitura warnings
import os
import numpy as np

from musicparser.data_loading import JTBDataModule
from musicparser.models import ArcPredictionLightModel

import wandb

download = False

if download:
    run = wandb.init()
    artifact = run.use_artifact('fosfrancesco/loo_JTB/model-7qr1vbyn:v0', type='model')
    artifact_dir = artifact.download()
else:
    artifact_dir = "artifacts/model-7qr1vbyn:v0"

datamodule = JTBDataModule(batch_size=1, num_workers=1, data_augmentation="preprocess", only_tree=True, tree_type="open", loo_index=8)
datamodule.setup()

pos_weight = int(datamodule.positive_weight)
print("Using pos_weight", pos_weight)
input_dim = 64
model = ArcPredictionLightModel.load_from_checkpoint(checkpoint_path=os.path.join(os.path.normpath(artifact_dir), "model.ckpt"))

wandb_logger = True

trainer = Trainer(
    max_epochs=60, accelerator="auto", devices= [0], #strategy="ddp",
    num_sanity_val_steps=1,
    logger=wandb_logger,
    deterministic=True
    )

# trainer.tune(model, datamodule=datamodule)
# print("LR set to", model.lr)
trainer.test(model,datamodule)
out_dict = trainer.predict(model, dataloaders=datamodule.test_dataloader())
# print(out_dict)
# print(np.mean(out_dict))