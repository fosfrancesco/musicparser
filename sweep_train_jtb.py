from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
import torch
import random
import argparse
import warnings
import wandb
warnings.filterwarnings('ignore') # avoid printing the partitura warnings

from musicparser.data_loading import JTBDataModule
from musicparser.models import ArcPredictionLightModel

torch.multiprocessing.set_sharing_strategy('file_system')

# for repeatability
seed_everything(0,workers=True)

wandb_run = wandb.init(group = "Sweep-JTB", job_type="JTB")
# Config parameters are automatically set by W&B sweep agent
config = wandb.config


# CUDA_VISIBLE_DEVICES=0 wandb agent fosfrancesco/sweeps_JTB/v00vqs1d
# CUDA_VISIBLE_DEVICES=1 wandb agent fosfrancesco/sweeps_JTB/v00vqs1d
# CUDA_VISIBLE_DEVICES=2 wandb agent fosfrancesco/sweeps_JTB/v00vqs1d
# CUDA_VISIBLE_DEVICES=3 wandb agent fosfrancesco/sweeps_JTB/v00vqs1d

###### Sweep optimizers ######

# CUDA_VISIBLE_DEVICES=0 wandb agent fosfrancesco/sweeps_JTB/ult4ybsb
# CUDA_VISIBLE_DEVICES=1 wandb agent fosfrancesco/sweeps_JTB/ult4ybsb
# CUDA_VISIBLE_DEVICES=2 wandb agent fosfrancesco/sweeps_JTB/ult4ybsb
# CUDA_VISIBLE_DEVICES=3 wandb agent fosfrancesco/sweeps_JTB/ult4ybsb

###### Sweep 2 ######
# CUDA_VISIBLE_DEVICES=0 wandb agent fosfrancesco/sweeps_JTB/p6ogzds5
# CUDA_VISIBLE_DEVICES=1 wandb agent fosfrancesco/sweeps_JTB/p6ogzds5
# CUDA_VISIBLE_DEVICES=2 wandb agent fosfrancesco/sweeps_JTB/p6ogzds5
# CUDA_VISIBLE_DEVICES=3 wandb agent fosfrancesco/sweeps_JTB/p6ogzds5


def main(config):
    # set parameters from config
    n_layers = config["n_layers"]
    n_hidden = config["n_hidden"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    dropout = config["dropout"]
    activation = config["activation"]
    use_pos_weight = config["use_pos_weight"]
    biaffine = config["biaffine"]
    encoder_type = config["encoder_type"]
    n_heads = config["n_heads"]
    loss_type = config["loss_type"]
    optimizer = config["optimizer"]
    warmup_steps = config["warmup_steps"]
    emb_arg = eval(config["embeddings"])
    if emb_arg == []:
        use_embeddings = False
        embedding_dim = {}
        emb_str = "noEmb"
    elif len(emb_arg) == 1:
        embedding_dim = {"sum": emb_arg[0]}
        use_embeddings = True
        emb_str = f"sum{emb_arg[0]}"
    else:
        embedding_dim = {"root": emb_arg[0], "form": emb_arg[1], "ext": emb_arg[2], "duration": emb_arg[3], "metrical" : emb_arg[4]} # sum roughtly 1/4 of the hidden size
        use_embeddings = True
        emb_str = f"r{emb_arg[0]}f{emb_arg[0]}e{emb_arg[0]}d{emb_arg[3]}m{emb_arg[4]}"
    
    rpr = "relative"
    pretrain = False
    num_workers = 20
    devices = [0]
    wandb_log = True
    patience = 30
    data_augmentation = "preprocess"

    datamodule = JTBDataModule(batch_size=1, num_workers=num_workers, data_augmentation=data_augmentation, only_tree=not pretrain)
    if use_pos_weight:
        pos_weight = int(datamodule.positive_weight)
        print("Using pos_weight", pos_weight)
    else:
        pos_weight = 1
    input_dim = sum(embedding_dim.values()) if use_embeddings else 25
    model = ArcPredictionLightModel(input_dim, n_hidden,pos_weight=pos_weight, dropout=dropout, lr=lr, weight_decay=weight_decay, n_layers=n_layers, activation=activation, use_embeddings=use_embeddings, embedding_dim=embedding_dim, biaffine=biaffine, encoder_type=encoder_type, n_heads=n_heads, data_type="chords", rpr = rpr, pretrain_mode= pretrain, loss_type = loss_type, optimizer=optimizer, warmup_steps=warmup_steps )

    if wandb_log:
        name = f"{encoder_type}-{n_layers}-{n_hidden}-lr={lr}-wd={weight_decay}-dr={dropout}-act={activation}-emb={emb_str}-aug={data_augmentation}-biaf={biaffine}-heads={n_heads}-rpr={rpr}-loss={loss_type}-PW={use_pos_weight}"        
        wandb_logger = WandbLogger(log_model = True, project="Parsing JTB", name= name )
    else:
        wandb_logger = True

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_ctree_sim", mode="max")
    early_stop_callback = EarlyStopping(monitor="val_ctree_sim", min_delta=0.00, patience=patience, verbose=True, mode="max")
    
    trainer = Trainer(
        max_epochs=100, accelerator="auto", devices= devices, #strategy="ddp",
        num_sanity_val_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=True,
        reload_dataloaders_every_n_epochs= 1 if data_augmentation=="online" else 0,
        )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule,ckpt_path=checkpoint_callback.best_model_path)



if __name__ == "__main__":
    print(f'Starting a run with {config}')
    main(config)