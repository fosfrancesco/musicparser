from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

from musicparser.data_loading import TSDataModule
from musicparser.models import ArcPredictionLightModel

num_workers = 20
embed_type = "RNN"
n_layers = 2
n_hidden = 300
lr = 0.005
weight_decay = 0.004
dropout = 0.3
wandb_log = False
patience = 10
devices = [1]
use_pos_weight = True
activation = "relu"

def main():
    datamodule = TSDataModule(batch_size=1, num_workers=num_workers)
    if use_pos_weight:
        pos_weight = int(datamodule.positive_weight)
        print("Using pos_weight", pos_weight)
    else:
        pos_weight = 1
    model = ArcPredictionLightModel(24, n_hidden,pos_weight=pos_weight, dropout=dropout, lr=lr, weight_decay=weight_decay, n_layers=n_layers, activation=activation)

    if wandb_log:
        # name = f"{n_layer}-{n_hidden}-{jk_mode}-{pot_edges_dist}-{linear_assignment}-{conv_type}-d{dropout}"
        name = f"{embed_type}-{n_layers}-{n_hidden}-lr={lr}-wd={weight_decay}-dr={dropout}-act={activation}"        
        wandb_logger = WandbLogger(log_model = True, project="Parsing TS", name= name )
    else:
        wandb_logger = True

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_fscore", mode="max")
    early_stop_callback = EarlyStopping(monitor="val_fscore", min_delta=0.00, patience=patience, verbose=True, mode="max")
    trainer = Trainer(
        max_epochs=100, accelerator="auto", devices= devices, #strategy="ddp",
        num_sanity_val_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        )

    trainer.fit(model, datamodule)



if __name__ == '__main__':
    main()