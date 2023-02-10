from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

from musicparser.data_loading import TSDataModule
from musicparser.models import ArcPredictionModel, ArcPredictionLightModel

num_workers = 4
embed_type = "RNN"
n_layers = 2
n_hidden = 300
lr = 0.001
weight_decay = 0.004
dropout = 0.3
reg_loss_weight = 1
wandb_log = False
patience = 10
devices = [0]

datamodule = TSDataModule(batch_size=1, num_workers=num_workers)
# datamodule.setup()
model = ArcPredictionLightModel(20, n_hidden,)

if wandb_log:
    # name = f"{n_layer}-{n_hidden}-{jk_mode}-{pot_edges_dist}-{linear_assignment}-{conv_type}-d{dropout}"
    name = "{}-{}x{}-{}-lr={}-wd={}-dr={}-rl={}-jk={}-act={}-ped={}".format(
        embed_type, n_layers, n_hidden, lr, weight_decay, dropout, reg_loss_weight)
    
    wandb_logger = WandbLogger(log_model = True, project="Parsing TS", name= name )
else:
    wandb_logger = True

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_fscore", mode="max")
early_stop_callback = EarlyStopping(monitor="val_fscore", min_delta=0.00, patience=patience, verbose=True, mode="max")
trainer = Trainer(
    max_epochs=100, accelerator="auto", #devices= devices, #strategy="ddp",
    num_sanity_val_steps=1,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stop_callback],
    )

trainer.fit(model, datamodule)



