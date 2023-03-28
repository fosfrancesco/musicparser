from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
import torch
import random
import argparse
import warnings
warnings.filterwarnings('ignore') # avoid printing the partitura warnings

from musicparser.data_loading import JTBDataModule
from musicparser.models import ArcPredictionLightModel

torch.multiprocessing.set_sharing_strategy('file_system')

# for repeatability
# torch.manual_seed(0)
# random.seed(0)
# torch.use_deterministic_algorithms(True)
seed_everything(0,workers=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="[2]")
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.21)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--wandb_log", action="store_true", help="Use wandb for logging.")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--data_augmentation", type=str, default="preprocess", help="'preprocess', 'no', or 'online'")
    parser.add_argument("--biaffine", action="store_true", help="Use biaffine arc decoder.")
    parser.add_argument("--pos_weight", action="store_true", help="Use positional weight on binary CE.")
    parser.add_argument('--encoder_type', type=str, default="transformer", help="'rnn', or 'transformer'")
    # parser.add_argument("--embeddings", type=str, default="[12,4,4,4,4]")
    parser.add_argument("--embeddings", type=str, default="[96]")
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--pos_enc', type= str, default="relative", help="'absolute', or 'relative'" )
    parser.add_argument('--pretrain', type= str, default="False", help="'True', or 'False'" )
    parser.add_argument('--loss', type= str, default="both", help="'bce', 'ce', or 'both'" )
    parser.add_argument('--optimizer', type= str, default="warmadamw", help="'adamw', 'radam', or 'warmadamw'" )
    parser.add_argument('--warmup_steps', type= int, default=50, help="warmup steps for warmadamw")
    parser.add_argument('--tree_type', type= str, default="open", help="'open' or 'complete'" )
    parser.add_argument('--max_epochs', type= int, default=50, help="max epochs for training")

    args = parser.parse_args()

    num_workers = args.num_workers
    n_layers = args.n_layers
    n_hidden = args.n_hidden
    lr = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    wandb_log = args.wandb_log
    patience = args.patience
    devices = eval(args.gpus)
    use_pos_weight = args.pos_weight
    activation = args.activation
    data_augmentation = args.data_augmentation
    biaffine = args.biaffine
    encoder_type = args.encoder_type
    n_heads = args.n_heads
    emb_arg = eval(args.embeddings)
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
    rpr = args.pos_enc == "relative"
    pretrain = eval(args.pretrain)
    loss_type = args.loss
    optimizer = args.optimizer
    warmup_steps = args.warmup_steps
    tree_type = args.tree_type
    max_epochs = args.max_epochs

    print("Starting a new run with the following parameters:")
    print(args)

    datamodule = JTBDataModule(batch_size=1, num_workers=num_workers, data_augmentation=data_augmentation, only_tree=not pretrain, tree_type=tree_type)
    datamodule.setup()
    if use_pos_weight:
        pos_weight = int(datamodule.positive_weight)
        print("Using pos_weight", pos_weight)
    else:
        pos_weight = 1
    input_dim = sum(embedding_dim.values()) if use_embeddings else 25
    model = ArcPredictionLightModel(input_dim, n_hidden,pos_weight=pos_weight, dropout=dropout, lr=lr, weight_decay=weight_decay, n_layers=n_layers, activation=activation, use_embeddings=use_embeddings, embedding_dim=embedding_dim, biaffine=biaffine, encoder_type=encoder_type, n_heads=n_heads, data_type="chords", rpr = rpr, pretrain_mode= pretrain, loss_type = loss_type, optimizer = optimizer, warmup_steps= warmup_steps, max_epochs = max_epochs, len_train_dataloader= len(datamodule.dataset_train))

    if wandb_log:
        name = f"{encoder_type}-{n_layers}-{n_hidden}-lr={lr}-wd={weight_decay}-dr={dropout}-act={activation}-emb={emb_str}-aug={data_augmentation}-biaf={biaffine}-heads={n_heads}-rpr={rpr}-loss={loss_type}-PW={use_pos_weight}-opt={optimizer}-warmup={warmup_steps}-T={tree_type}"        
        wandb_logger = WandbLogger(log_model = True, project="Parsing JTB", name= name )
    else:
        wandb_logger = True

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_head_accuracy_postp", mode="max")
    early_stop_callback = EarlyStopping(monitor="val_head_accuracy_postp", min_delta=0.00, patience=patience, verbose=True, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = Trainer(
        max_epochs=max_epochs, accelerator="auto", devices= devices, #strategy="ddp",
        num_sanity_val_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        deterministic=True,
        reload_dataloaders_every_n_epochs= 1 if data_augmentation=="online" else 0,
        log_every_n_steps=10
        )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule,ckpt_path=checkpoint_callback.best_model_path)



if __name__ == '__main__':
    main()