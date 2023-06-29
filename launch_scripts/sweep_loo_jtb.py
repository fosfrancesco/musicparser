from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
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

wandb_run = wandb.init(group = "LOO1", job_type="LOO1")
# Config parameters are automatically set by W&B sweep agent
config = wandb.config

def main(config):
    # set parameters from config
    loo_index = config["loo_index"]
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
    patience = 100
    data_augmentation = "preprocess"
    tree_type = "open"
    max_epochs = 60

    datamodule = JTBDataModule(batch_size=1, num_workers=num_workers, data_augmentation=data_augmentation, only_tree=not pretrain, tree_type = tree_type, loo_index=int(loo_index))
    datamodule.setup()
    if use_pos_weight:
        pos_weight = int(datamodule.positive_weight)
        print("Using pos_weight", pos_weight)
    else:
        pos_weight = 1
    input_dim = sum(embedding_dim.values()) if use_embeddings else 25
    model = ArcPredictionLightModel(input_dim, n_hidden,pos_weight=pos_weight, dropout=dropout, lr=lr, weight_decay=weight_decay, n_layers=n_layers, activation=activation, use_embeddings=use_embeddings, embedding_dim=embedding_dim, biaffine=biaffine, encoder_type=encoder_type, n_heads=n_heads, data_type="chords", rpr = rpr, pretrain_mode= pretrain, loss_type = loss_type, optimizer=optimizer, warmup_steps=warmup_steps, max_epochs = max_epochs, len_train_dataloader= len(datamodule.dataset_train) )

    
    name = ""
    wandb_logger = WandbLogger(log_model = True, project="Parsing JTB", name= name )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=max_epochs, accelerator="auto", devices= devices, #strategy="ddp",
        num_sanity_val_steps=1,
        logger=wandb_logger,
        callbacks=[lr_monitor],
        deterministic=True,
        reload_dataloaders_every_n_epochs= 1 if data_augmentation=="online" else 0,
        # log_every_n_steps=10
        )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)



if __name__ == "__main__":
    print(f'Starting a run with {config}')
    main(config)