from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
import torch
import random
import argparse
import warnings
import wandb
warnings.filterwarnings('ignore') # avoid printing the partitura warnings

from musicparser.data_loading import TSDataModule
from musicparser.models import ArcPredictionLightModel

torch.multiprocessing.set_sharing_strategy('file_system')

# for repeatability
torch.manual_seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)

wandb_run = wandb.init(group = "Sweep-TS", job_type="TS")
# Config parameters are automatically set by W&B sweep agent
config = wandb.config


# CUDA_VISIBLE_DEVICES=0 wandb agent fosfrancesco/sweep_TS/a8f6wlp4
# CUDA_VISIBLE_DEVICES=1 wandb agent fosfrancesco/sweep_TS/a8f6wlp4
# CUDA_VISIBLE_DEVICES=2 wandb agent fosfrancesco/sweep_TS/a8f6wlp4
# CUDA_VISIBLE_DEVICES=3 wandb agent fosfrancesco/sweep_TS/a8f6wlp4

# only transformer

# CUDA_VISIBLE_DEVICES=0 wandb agent fosfrancesco/sweep_TS/76m4zi32
# CUDA_VISIBLE_DEVICES=1 wandb agent fosfrancesco/sweep_TS/76m4zi32
# CUDA_VISIBLE_DEVICES=2 wandb agent fosfrancesco/sweep_TS/76m4zi32
# CUDA_VISIBLE_DEVICES=3 wandb agent fosfrancesco/sweep_TS/76m4zi32

def main(config):
    # set parameters from config
    n_layers = config["n_layers"]
    n_hidden = config["n_hidden"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    dropout = config["dropout"]
    activation = config["activation"]
    biaffine = config["biaffine"]
    encoder_type = config["encoder_type"]
    n_heads = config["n_heads"]
    emb_arg = eval(config["embeddings"])
    if emb_arg == []:
        use_embeddings = False
        embedding_dim = {}
        emb_str = "noEmb"
    else:
        embedding_dim = {"pitch": emb_arg[0], "duration": emb_arg[1], "metrical": emb_arg[2]} # sum roughtly 1/4 of the hidden size
        use_embeddings = True
        emb_str = f"p{emb_arg[0]}d{emb_arg[1]}m{emb_arg[2]}"

    num_workers = 20
    devices = [0]
    wandb_log = True
    patience = 15
    use_pos_weight = True
    data_augmentation = "preprocess"


    datamodule = TSDataModule(batch_size=1, num_workers=num_workers, will_use_embeddings=use_embeddings, data_augmentation=data_augmentation)
    if use_pos_weight:
        pos_weight = int(datamodule.positive_weight)
        print("Using pos_weight", pos_weight)
    else:
        pos_weight = 1
    input_dim = embedding_dim["pitch"] + embedding_dim["duration"] + embedding_dim["metrical"] if use_embeddings else 25
    model = ArcPredictionLightModel(input_dim, n_hidden,pos_weight=pos_weight, dropout=dropout, lr=lr, weight_decay=weight_decay, n_layers=n_layers, activation=activation, use_embeddings=use_embeddings, embedding_dim=embedding_dim, biaffine=biaffine, encoder_type=encoder_type, n_heads=n_heads)

    if wandb_log:
        name = f"{encoder_type}-{n_layers}-{n_hidden}-lr={lr}-wd={weight_decay}-dr={dropout}-act={activation}-emb={emb_str}-aug={data_augmentation}-biaf={biaffine}-heads={n_heads}"        
        wandb_logger = WandbLogger(log_model = True, project="Parsing TS", name= name )
    else:
        wandb_logger = True

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_fscore_postp", mode="max")
    early_stop_callback = EarlyStopping(monitor="val_fscore_postp", min_delta=0.00, patience=patience, verbose=True, mode="max")
    trainer = Trainer(
        max_epochs=200, accelerator="auto", devices= devices, #strategy="ddp",
        num_sanity_val_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)



if __name__ == "__main__":
    print(f'Starting a run with {config}')
    main(config)