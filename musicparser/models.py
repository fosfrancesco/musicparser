import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torch_geometric as pyg
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, MulticlassAccuracy
import numpy as np

from musicparser.rpr import TransformerEncoderLayerRPR, TransformerEncoderRPR, DummyDecoder
from musicparser.postprocessing import chuliu_edmonds_one_root
from musicparser.data_loading import DURATIONS, get_feats_one_hot, METRICAL_LEVELS, NUMBER_OF_PITCHES, CHORD_FORM, CHORD_EXTENSION, JTB_DURATION


class ArcPredictionLightModel(LightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers=2,
        activation="relu",
        dropout=0.3,
        lr=0.001,
        weight_decay=5e-4,
        pos_weight = None,
        embedding_dim = {"pitch": 24, "duration": 6, "metrical": 2},
        use_embeddings = True,
        biaffine = False,
        encoder_type = "rnn",
        n_heads = 4,
        data_type = "notes",
        rpr = False,
        pretrain_mode = False
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.module = ArcPredictionModel(
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            embedding_dim,
            use_embeddings,
            biaffine,
            encoder_type,
            n_heads,
            data_type,
            rpr,
            pretrain_mode
        )
        pos_weight = 1 if pos_weight is None else pos_weight
        self.train_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))
        self.val_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))
        self.val_f1score = BinaryF1Score()
        self.val_f1score_postp = BinaryF1Score()
        self.val_accuracy = BinaryAccuracy()
        self.test_f1score = BinaryF1Score()
        self.test_f1score_postp = BinaryF1Score()
        self.pretrain_mode = pretrain_mode
        if pretrain_mode:
            # self.pre_train_loss = nn.ModuleDict({"root": CrossEntropyLoss(), "form": CrossEntropyLoss(), "ext": CrossEntropyLoss(), "dur": CrossEntropyLoss(), "met": CrossEntropyLoss()})
            self.pre_train_loss = CrossEntropyLoss()
            self.pre_train_accuracy = nn.ModuleDict({"root": MulticlassAccuracy(12), "form": MulticlassAccuracy(len(CHORD_FORM)), "ext": MulticlassAccuracy(len(CHORD_EXTENSION)), "dur": MulticlassAccuracy(len(JTB_DURATION)), "met": MulticlassAccuracy(METRICAL_LEVELS)})
            # self.pre_val_loss = nn.ModuleDict({"root": CrossEntropyLoss(), "form": CrossEntropyLoss(), "ext": CrossEntropyLoss(), "dur": CrossEntropyLoss(), "met": CrossEntropyLoss()})
            self.pre_val_loss = CrossEntropyLoss()
            self.pre_val_accuracy = nn.ModuleDict({"root": MulticlassAccuracy(12), "form": MulticlassAccuracy(len(CHORD_FORM)), "ext": MulticlassAccuracy(len(CHORD_EXTENSION)), "dur": MulticlassAccuracy(len(JTB_DURATION)), "met": MulticlassAccuracy(METRICAL_LEVELS)})

    def training_step(self, batch, batch_idx):
        note_seq, truth_arcs_mask, pot_arcs = batch
        note_seq, truth_arcs_mask, pot_arcs = note_seq[0], truth_arcs_mask[0], pot_arcs[0]
        if not self.pretrain_mode: # normal mode, predict arcs
            arc_pred_mask_logits = self.module(note_seq, pot_arcs)
            loss = self.train_loss(arc_pred_mask_logits.float(), truth_arcs_mask.float()).cpu()
            self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            return loss
        else: # pretrain mode, predict chord labels
            # shift input sequence to the right, and shorten prediction by one, to compare with prediction at next position
            input = note_seq[:-1,:]
            expected = note_seq[1:,:]
            # get mask for input sequence
            mask = generate_square_subsequent_mask(len(expected)).to(self.device)
            # predict chord labels
            pred_logits = self.module(input,None,mask=mask)
            loss = 0
            accuracy = 0
            for i,key in enumerate(pred_logits.keys()):
                loss+= self.pre_train_loss(pred_logits[key], expected[:,i].long())
                # self.log(f"train_loss_{key}", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
                accuracy += self.pre_train_accuracy[key](pred_logits[key], expected[:,i].long())
                # self.log(f"train_acc_{key}", acc.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            # loss = loss/len(pred_logits.keys())
            accuracy = accuracy/len(pred_logits.keys())
            self.log("pre_train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            self.log("pre_train_acc", accuracy.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            return loss


    def validation_step(self, batch, batch_idx):
        note_seq, truth_arcs_mask, pot_arcs = batch
        note_seq, truth_arcs_mask, pot_arcs = note_seq[0], truth_arcs_mask[0], pot_arcs[0]
        if not self.pretrain_mode: # normal mode, predict arcs
            num_notes = len(note_seq)
            arc_pred_mask_logits = self.module(note_seq, pot_arcs)
            loss = self.val_loss(arc_pred_mask_logits.float(), truth_arcs_mask.float()).cpu()
            self.log("val_loss", loss.item(), on_epoch=True, batch_size=1)

            arc_pred__mask_normalized = torch.sigmoid(arc_pred_mask_logits)
            pred_arc = pot_arcs[torch.round(arc_pred__mask_normalized).squeeze().bool()]
            # compute pred and ground truth adj matrices
            if torch.sum(pred_arc) > 0:
                adj_pred = pyg.utils.to_dense_adj(pred_arc.T, max_num_nodes=num_notes).squeeze().cpu()
            else: # to avoid exception in to_dense_adj when there is no predicted edge
                adj_pred = torch.zeros((num_notes, num_notes)).squeeze().to(self.device).cpu()
            ## compute loss and F1 score
            adj_target = pyg.utils.to_dense_adj(pot_arcs[truth_arcs_mask].T, max_num_nodes=num_notes).squeeze().long().cpu()
            val_fscore = self.val_f1score.cpu()(adj_pred.flatten(), adj_target.flatten())
            self.log("val_fscore", val_fscore.item(), prog_bar=True, batch_size=1)
            val_accuracy = self.val_accuracy.cpu()(adj_pred.flatten(), adj_target.flatten())
            self.log("val_accuracy", val_accuracy.item(), prog_bar=True, batch_size=1)
            # postprocess with chuliu edmonds algorithm https://wendy-xiao.github.io/posts/2020-07-10-chuliuemdond_algorithm/ 
            adj_pred_probs = torch.sparse_coo_tensor(pot_arcs.T, arc_pred__mask_normalized, (num_notes, num_notes)).cpu().to_dense().numpy()
            # add a new upper row and left column for the root to the adjency matrix
            adj_pred_probs_root = np.vstack((np.zeros((1, num_notes)), adj_pred_probs))
            adj_pred_probs_root = np.hstack((np.zeros((num_notes+1, 1)), adj_pred_probs_root))
            # transpose to have an adjency matrix with edges pointing toward the parent node and take log probs
            adj_pred_log_probs_transp_root = np.log(adj_pred_probs_root.T)
            # postprocess with chu-liu edmonds algorithm
            head_seq = chuliu_edmonds_one_root(adj_pred_log_probs_transp_root)
            head_seq = head_seq[1:] # remove the root
            # structure the postprocess results in an adjency matrix with edges that point toward the child node
            adj_pred_postp = torch.zeros((num_notes,num_notes))
            for i, head in enumerate(head_seq):
                if head != 0:
                    # id is index in note list + 1
                    adj_pred_postp[head-1, i] = 1
                else: #handle the root
                    root = i
            val_fscore_postp = self.val_f1score_postp.cpu()(adj_pred_postp.flatten(), adj_target.flatten())
            self.log("val_fscore_postp", val_fscore_postp.item(), prog_bar=True, batch_size=1)
        else:
            # shift input sequence to the right, and shorten prediction by one, to compare with prediction at next position
            input = note_seq[:-1,:]
            expected = note_seq[1:,:]
            # get mask for input sequence
            mask = generate_square_subsequent_mask(len(expected)).to(self.device)
            # predict chord labels
            pred_logits = self.module(input,None,mask=mask)
            loss = 0
            accuracy = 0
            for i,key in enumerate(pred_logits.keys()):
                loss+= self.pre_train_loss(pred_logits[key], expected[:,i].long())
                # self.log(f"train_loss_{key}", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
                accuracy += self.pre_train_accuracy[key](pred_logits[key], expected[:,i].long())
                # self.log(f"train_acc_{key}", acc.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            # loss = loss/len(pred_logits.keys())
            accuracy = accuracy/len(pred_logits.keys())
            self.log("pre_val_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=1)
            self.log("pre_val_acc", accuracy.item(), prog_bar=True, on_epoch=True, batch_size=1)

    
    def test_step(self, batch, batch_idx):
        note_seq, truth_arcs_mask, pot_arcs = batch
        note_seq, truth_arcs_mask, pot_arcs = note_seq[0], truth_arcs_mask[0], pot_arcs[0]
        num_notes = len(note_seq)
        arc_pred_mask_logits = self.module(note_seq, pot_arcs)
        arc_pred__mask_normalized = torch.sigmoid(arc_pred_mask_logits)
        pred_arc = pot_arcs[torch.round(arc_pred__mask_normalized).squeeze().bool()]
        # compute pred and ground truth adj matrices
        if torch.sum(pred_arc) > 0:
            adj_pred = pyg.utils.to_dense_adj(pred_arc.T, max_num_nodes=num_notes).squeeze().cpu()
        else: # to avoid exception in to_dense_adj when there is no predicted edge
            adj_pred = torch.zeros((num_notes, num_notes)).squeeze().to(self.device).cpu()
        ## compute loss and F1 score
        adj_target = pyg.utils.to_dense_adj(pot_arcs[truth_arcs_mask].T, max_num_nodes=num_notes).squeeze().long().cpu()
        test_fscore = self.test_f1score.cpu()(adj_pred.flatten(), adj_target.flatten())
        self.log("test_fscore", test_fscore.item(), prog_bar=True, batch_size=1)
        # postprocess with chuliu edmonds algorithm https://wendy-xiao.github.io/posts/2020-07-10-chuliuemdond_algorithm/ 
        adj_pred_probs = torch.sparse_coo_tensor(pot_arcs.T, arc_pred__mask_normalized, (num_notes, num_notes)).cpu().to_dense().numpy()
        # add a new upper row and left column for the root to the adjency matrix
        adj_pred_probs_root = np.vstack((np.zeros((1, num_notes)), adj_pred_probs))
        adj_pred_probs_root = np.hstack((np.zeros((num_notes+1, 1)), adj_pred_probs_root))
        # transpose to have an adjency matrix with edges pointing toward the parent node and take log probs
        adj_pred_log_probs_transp_root = np.log(adj_pred_probs_root.T)
        # postprocess with chu-liu edmonds algorithm
        head_seq = chuliu_edmonds_one_root(adj_pred_log_probs_transp_root)
        head_seq = head_seq[1:] # remove the root
        # structure the postprocess results in an adjency matrix with edges that point toward the child node
        adj_pred_postp = torch.zeros((num_notes,num_notes))
        for i, head in enumerate(head_seq):
            if head != 0:
                # id is index in note list + 1
                adj_pred_postp[head-1, i] = 1
            else: #handle the root
                root = i
        test_fscore_postp = self.test_f1score_postp.cpu()(adj_pred_postp.flatten(), adj_target.flatten())
        self.log("test_fscore_postp", test_fscore_postp.item(), prog_bar=True, batch_size=1)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, verbose=True)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }



class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        encoder_depth,
        n_heads = 4,
        dropout=None,
        activation = "relu",
        rpr = False
    ):
        super().__init__()

        if dropout is None:
            dropout = 0
        self.input_dim = input_dim
        self.rpr = rpr

        self.positional_encoder = PositionalEncoding(
            d_model=input_dim, dropout=dropout, max_len=200
        )
        # self.dummy = DummyDecoder()
        if not rpr: # normal transformer with absolute positional representation
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            # self.transformer = nn.Transformer(
            #     d_model=input_dim, nhead=n_heads, num_encoder_layers=encoder_depth,
            #     num_decoder_layers=0, dropout=dropout, activation=activation,
            #     dim_feedforward=hidden_dim, custom_decoder=self.dummy
            # )
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, dim_feedforward=hidden_dim, nhead=n_heads, dropout =dropout, activation=activation)
            encoder_norm = nn.LayerNorm(input_dim)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth, norm=encoder_norm)
        else: # relative positional representation
            encoder_norm = nn.LayerNorm(input_dim)
            encoder_layer = TransformerEncoderLayerRPR(input_dim, n_heads, hidden_dim, dropout, activation=activation, er_len=200)
            self.transformer_encoder = TransformerEncoderRPR(encoder_layer, encoder_depth, encoder_norm)
            # self.transformer = nn.Transformer(
            #     d_model=input_dim, nhead=n_heads, num_encoder_layers=encoder_depth,
            #     num_decoder_layers=0, dropout=dropout, activation=activation,
            #     dim_feedforward=hidden_dim, custom_decoder=self.dummy, custom_encoder=encoder
            # )

    def forward(self, z, src_mask=None):
        # TODO: why this is rescaled like that?
        # add positional encoding
        z = self.positional_encoder(z)
        # reshape to (seq_len, batch = 1, input_dim)
        z = torch.unsqueeze(z,dim= 1)
        # run transformer encoder
        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        z = self.transformer_encoder(src=z, mask=src_mask)
        # remove batch dim
        z = torch.squeeze(z, dim=1)
        return z, ""


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class NotesEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        rnn_depth,
        dropout=0.1,
        embedding_dim = {},
        use_embeddings = True,
        encoder_type = "rnn",
        bidirectional=True,
        activation = "relu",
        n_heads = 4,
        data_type = "notes",
        rpr = False,
    ):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if use_embeddings and embedding_dim == {}:
            raise ValueError("If use_embeddings is True, embedding_dim must be provided")
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.use_embeddings = use_embeddings
        self.data_type = data_type
        self.embedding_dim = embedding_dim

        # Encoder layer
        if encoder_type == "rnn":
            self.encoder_cell = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
                bidirectional=bidirectional,
                num_layers=rnn_depth,
                dropout=dropout,
            )
        elif encoder_type == "transformer":
            self.encoder_cell = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                encoder_depth=rnn_depth,
                dropout=dropout,
                activation=activation,
                n_heads=n_heads,
                rpr = rpr
            )
        else:
            raise ValueError(f"Encoder type {encoder_type} not supported")
        # embedding layer
        if use_embeddings:
            if data_type == "notes":
                if not "sum" in embedding_dim.keys():
                    self.embeddings = nn.ModuleDict({
                        "pitch": nn.Embedding(NUMBER_OF_PITCHES, embedding_dim["pitch"]),
                        "duration": nn.Embedding(len(DURATIONS), embedding_dim["duration"]),
                        "metrical": nn.Embedding(METRICAL_LEVELS, embedding_dim["metrical"])
                    })
                else:
                    self.embeddings = nn.ModuleDict({
                        "pitch": nn.Embedding(NUMBER_OF_PITCHES, embedding_dim["sum"]),
                        "duration": nn.Embedding(len(DURATIONS), embedding_dim["sum"]),
                        "metrical": nn.Embedding(METRICAL_LEVELS, embedding_dim["sum"])
                    })

            elif data_type == "chords":
                # root_numbers, chord_forms, chord_extensions, duration_indices, metrical_indices
                if not "sum" in embedding_dim.keys():
                    self.embeddings = nn.ModuleDict({
                        "root": nn.Embedding(12, embedding_dim["root"]),
                        "form": nn.Embedding(len(CHORD_FORM), embedding_dim["form"]),
                        "ext": nn.Embedding(len(CHORD_EXTENSION), embedding_dim["ext"]),
                        "duration": nn.Embedding(len(JTB_DURATION), embedding_dim["duration"]),
                        "metrical": nn.Embedding(METRICAL_LEVELS, embedding_dim["metrical"])
                    })
                else:
                    self.embeddings = nn.ModuleDict({
                        "root": nn.Embedding(12, embedding_dim["sum"]),
                        "form": nn.Embedding(len(CHORD_FORM), embedding_dim["sum"]),
                        "ext": nn.Embedding(len(CHORD_EXTENSION), embedding_dim["sum"]),
                        "duration": nn.Embedding(len(JTB_DURATION), embedding_dim["sum"]),
                        "metrical": nn.Embedding(METRICAL_LEVELS, embedding_dim["sum"])
                    })
            else:
                raise ValueError(f"Data type {data_type} not supported")

    def forward(self, sequence, mask=None):
        if self.use_embeddings:
            # run embedding
            if self.data_type == "notes":
                # we are discarding rests information at [:,1] because it is in the pitch
                pitch = sequence[:,0]
                duration = sequence[:,2]
                metrical = sequence[:,3]
                pitch = self.embeddings["pitch"](pitch.long())
                duration = self.embeddings["duration"](duration.long())
                metrical = self.embeddings["metrical"](metrical.long())
                if not "sum" in self.embedding_dim.keys():
                    # concatenate embeddings
                    z = torch.hstack((pitch, duration, metrical))
                else:
                    # sum all embeddings
                    z = pitch + duration + metrical
            elif self.data_type == "chords":
                root = sequence[:,0]
                form = sequence[:,1]
                ext = sequence[:,2]
                duration = sequence[:,3]
                metrical = sequence[:,4]
                root = self.embeddings["root"](root.long())
                form = self.embeddings["form"](form.long())
                ext = self.embeddings["ext"](ext.long())
                duration = self.embeddings["duration"](duration.long())
                metrical = self.embeddings["metrical"](metrical.long())
                if not "sum" in self.embedding_dim.keys():
                    # concatenate embeddings
                    z = torch.hstack((root, form, ext, duration, metrical))
                else:
                    # sum all embeddings
                    z = root + form + ext + duration + metrical
        else:
            # one hot encoding
            z = get_feats_one_hot(sequence)

        if mask is None:
            z, _ = self.encoder_cell(z)
        else:
            z, _ = self.encoder_cell(z, src_mask= mask)

        # if self.dropout is not None:
        z = self.dropout(z)
        return z
      

class ArcDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, activation=F.relu, dropout=0.3, biaffine=True, pretrain_mode = False):
        super().__init__()
        self.activation = activation
        self.biaffine = biaffine
        self.pretrain_mode = pretrain_mode
        if not pretrain_mode: # normal functioning, predicting arcs
            if biaffine:
                self.lin1 = nn.Linear(hidden_channels, hidden_channels)
                self.lin2 = nn.Linear(hidden_channels, hidden_channels)
                self.bilinear = nn.Bilinear(hidden_channels , hidden_channels, 1)
            else:
                self.lin1 = nn.Linear(2*hidden_channels, hidden_channels)
                self.lin2 = nn.Linear(hidden_channels, 1)
        else: # pretraining mode, predicting chords
            self.lin_root = nn.Linear(hidden_channels, 12)
            self.lin_form = nn.Linear(hidden_channels, len(CHORD_FORM))
            self.lin_ext = nn.Linear(hidden_channels, len(CHORD_EXTENSION))
            self.lin_duration = nn.Linear(hidden_channels, len(JTB_DURATION))
            self.lin_metrical = nn.Linear(hidden_channels, METRICAL_LEVELS)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_channels)
        

    def forward(self, z, pot_arcs):
        z = self.norm(z)
        if not self.pretrain_mode: # normal functioning, predicting arcs
            if self.biaffine:
                # get the embeddings of the starting and ending nodes, both of shape (num_pot_arcs, hidden_channels)
                input1 =  z[pot_arcs[:, 0]]
                input2 = z[pot_arcs[:, 1]]
                # pass through a linear layer, shape (num_pot_arcs, hidden_channels)
                input1 = self.lin1(input1)
                input2 = self.lin2(input2)
                # pass through an activation function, shape (num_pot_arcs, hidden_channels)
                input1 = self.activation(input1)
                input2 = self.activation(input2)
                # normalize
                input1 = self.norm(input1)
                input2 = self.norm(input2)
                # pass through a dropout layer, shape (num_pot_arcs, hidden_channels)
                input1 = self.dropout(input1)
                input2 = self.dropout(input2)
                # # concatenate, like it is done in the stanza parser
                # input1 =  torch.cat((input1, torch.ones((input1.shape[0],1), device = input1.device)), dim = -1)
                # input2 = torch.cat((input2, torch.ones((input1.shape[0],1), device = input1.device)), dim = -1)
                z = self.bilinear(input1, input2)
            else:
                # concat the embeddings of the two nodes, shape (num_pot_arcs, 2*hidden_channels)
                z = torch.cat([z[pot_arcs[:, 0]], z[pot_arcs[:, 1]]], dim=-1)
                # pass through a linear layer, shape (num_pot_arcs, hidden_channels)
                z = self.lin1(z)
                # pass through activation, shape (num_pot_arcs, hidden_channels)
                z = self.activation(z)
                # normalize
                z = self.norm(z)
                # dropout
                z = self.dropout(z)
                # pass through another linear layer, shape (num_pot_arcs, 1)
                z = self.lin2(z)
            # return a vector of shape (num_pot_arcs,)
            return z.view(-1)
        else: # pretraining mode, predicting chords
            out = {}
            out["root"] = self.lin_root(z)
            out["form"] = self.lin_form(z)
            out["ext"] = self.lin_ext(z)
            out["dur"] = self.lin_duration(z)
            out["met"] = self.lin_metrical(z)
            return out


class ArcPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation="relu", dropout=0.2, embedding_dim = {}, use_embedding = True, biaffine = False, encoder_type = "rnn", n_heads = 4, data_type = "notes", rpr = False, pretrain_mode = False):
        super().__init__()
        if activation == "relu":
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise ValueError("Unknown activation function")
        self.activation = activation
        # initialize the encoder
        self.encoder = NotesEncoder(input_dim, hidden_dim, num_layers, dropout, embedding_dim, use_embedding, encoder_type, activation=activation, n_heads=n_heads, data_type = data_type, rpr =rpr)
        # set the dimension that the decoder will expect
        if encoder_type == "rnn":
            self.decoder_dim = hidden_dim
        else: # transformer case, the hidden dim is the dimension of the input, i.e., the embeddings
            self.decoder_dim = input_dim
        self.pretrain_mode = pretrain_mode
        # initialize the decoder
        self.decoder = ArcDecoder(self.decoder_dim, activation=activation, dropout=dropout, biaffine=biaffine, pretrain_mode=pretrain_mode)

    def forward(self, note_features, pot_arcs, mask=None):
        z = self.encoder(note_features, mask)
        return self.decoder(z, pot_arcs)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

