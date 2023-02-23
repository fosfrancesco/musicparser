from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

from musicparser.data_loading import TSDataModule
from musicparser.models import ArcPredictionLightModel

num_workers = 20
n_layers = 2
n_hidden = 128
lr = 0.005
weight_decay = 0.004
dropout = 0.1
wandb_log = False
patience = 30
devices = [0]
use_pos_weight = True
activation = "relu"
data_augmentation = True
embedding_dim = {"pitch": 20, "duration": 8, "metrical": 4} # roughtly 1/4 of the hidden size
use_embeddings = True
biaffine = False
encoder_type = "transformer"


def main():
    datamodule = TSDataModule(batch_size=1, num_workers=num_workers, will_use_embeddings=use_embeddings, data_augmentation=data_augmentation)
    if use_pos_weight:
        pos_weight = int(datamodule.positive_weight)
        print("Using pos_weight", pos_weight)
    else:
        pos_weight = 1
    input_dim = embedding_dim["pitch"] + embedding_dim["duration"] + embedding_dim["metrical"] if use_embeddings else 25
    model = ArcPredictionLightModel(input_dim, n_hidden,pos_weight=pos_weight, dropout=dropout, lr=lr, weight_decay=weight_decay, n_layers=n_layers, activation=activation, use_embeddings=use_embeddings, embedding_dim=embedding_dim, biaffine=biaffine, encoder_type=encoder_type)

    if wandb_log:
        name = f"{encoder_type}-{n_layers}-{n_hidden}-lr={lr}-wd={weight_decay}-dr={dropout}-act={activation}"        
        wandb_logger = WandbLogger(log_model = True, project="Parsing TS", name= name )
    else:
        wandb_logger = True

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_fscore", mode="max")
    early_stop_callback = EarlyStopping(monitor="val_fscore", min_delta=0.00, patience=patience, verbose=True, mode="max")
    trainer = Trainer(
        max_epochs=200, accelerator="auto", devices= devices, #strategy="ddp",
        num_sanity_val_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        )

    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule)



if __name__ == '__main__':
    main()