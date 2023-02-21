import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torch_geometric as pyg
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import numpy as np

from musicparser.postprocessing import chuliu_edmonds_one_root
from musicparser.data_loading import DURATIONS, get_feats_one_hot




class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        encoder_depth,
        n_heads = 8,
        dropout=None,
        activation = "relu"
    ):
        super().__init__()

        if dropout is None:
            dropout = 0

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, dim_feedforward=hidden_dim, nhead=n_heads, dropout =dropout, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

    def forward(self, sentences, sentences_len=None):
        z = self.self.transformer_encoder(sentences)
        return z


class RNNEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        rnn_depth,
        dropout=0,
        embedding_dim = {},
        use_embeddings = True,
        bidirectional=True,
    ):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        if use_embeddings and embedding_dim == {}:
            raise ValueError("If use_embeddings is True, embedding_dim must be provided")
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.use_embeddings = use_embeddings

        # RNN layer.
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
            dropout=dropout,
        )
        # embedding layer
        if use_embeddings:
            self.embedding_dim = 128 + len(DURATIONS)+ 1
            self.embedding_pitch = nn.Embedding(128, embedding_dim["pitch"])
            self.embedding_duration = nn.Embedding(len(DURATIONS), embedding_dim["duration"])
            self.embedding_metrical = nn.Embedding(2, embedding_dim["metrical"])

    def forward(self, sequence, sentences_len=None):
        pitch = sequence[:,0]
        is_rest = sequence[:,1]
        duration = sequence[:,2]
        metrical = sequence[:,3]
        if self.use_embeddings:
            # run embedding 
            pitch = self.embedding_pitch(pitch.long())
            duration = self.embedding_duration(duration.long())
            metrical = self.embedding_metrical(metrical.long())
            # concatenate embeddings
            z = torch.hstack((pitch, duration, metrical))
        else:
            # everything is already built in the data preparation function
            z = get_feats_one_hot(sequence).double()
        z, _ = self.rnn(z)
        # rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        # if self.dropout is not None:
        z = self.dropout(z)
        return z


# class GNNEncoder(torch.nn.Module):
# 	def __init__(self, input_channels, hidden_channels, out_channels, num_layers=1, activation=F.relu, dropout=0.5):
# 		super().__init__()
# 		self.num_layers = num_layers
# 		self.conv_layers = nn.ModuleList()
# 		for _ in range(num_layers - 1):
# 			conv = get_conv_from_string(conv_type, input_channels, hidden_channels, metadata) 
# 			self.conv_layers.append(conv)
# 		conv = get_conv_from_string(conv_type, input_channels, hidden_channels, metadata)
# 		self.conv_layers.append(conv)
# 		self.metadata = metadata
# 		self.normalize = gnn.GraphNorm(hidden_channels)
# 		self.dropout = nn.Dropout(dropout)
# 		self.activation = activation
# 		# self.first_linear = gnn.Linear(-1, hidden_channels)
# 		if jk_mode is not None:
# 			self.jk = gnn.JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers+1)
# 		else:
# 			self.jk = None
# 		self.conv_type = conv_type

# 	def forward(self, x, edge_index):
# 		if self.conv_type in ['HGTConv', 'HEATConv','HANConv']: # some models are inherently heterogenous
# 			h_dict = x
# 			# h_dict = {key: self.first_linear(h) for key, h in h_dict.items()}
# 			# h_dict = {key: self.activation(h) for key, h in h_dict.items()}
# 			# h_dict = {key: self.normalize(h) for key, h in h_dict.items()}
# 			for conv in self.conv_layers:
# 				h_dict = conv(h_dict, edge_index)
# 				# h_dict = {key: self.activation(h) for key, h in h_dict.items()}
# 				# h_dict = {key: self.normalize(h) for key, h in h_dict.items()}
# 				h_dict = {key: self.dropout(h) for key, h in h_dict.items()}
# 			return h_dict
# 		else:
# 			hs = list() # to save hidden stated for jump connection
# 			h = x
# 			for conv in self.conv_layers:
# 				h = conv(h, edge_index)
# 				h = self.activation(h)
# 				h = self.normalize(h)
# 				h = self.dropout(h)
# 				hs.append(h)
# 			# h = self.conv_layers[-1](h, edge_index)
# 			# hs.append(h)
# 			if self.jk is not None:
# 				h = self.jk(hs)

# 			return h


class ArcDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, activation=F.relu, dropout=0.3, biaffine=True):
        super().__init__()
        self.activation = activation
        self.biaffine = biaffine
        if biaffine:
            self.lin1 = nn.Linear(hidden_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, hidden_channels)
            self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, 1)
        else:
            self.lin1 = nn.Linear(2*hidden_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, pot_arcs):
        if self.biaffine:
            input1 = z[pot_arcs[:, 0]]
            input2 = z[pot_arcs[:, 1]]
            z = self.bilinear(self.dropout(self.activation(self.lin1(input1))), self.dropout(self.activation(self.lin2(input2))))
        else:
            z = torch.cat([z[pot_arcs[:, 0]], z[pot_arcs[:, 1]]], dim=-1)
            z = self.lin1(z)
            z = self.activation(z)
            z = self.lin2(z)
        return z.view(-1)


class ArcPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation="relu", dropout=0.2, embedding_dim = {}, use_embedding = True, biaffine = True):
        super().__init__()
        if activation == "relu":
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        self.encoder = RNNEncoder(input_dim, hidden_dim, num_layers, dropout, embedding_dim, use_embedding)
        self.decoder = ArcDecoder(hidden_dim, activation=activation, dropout=dropout, biaffine=biaffine)

    def forward(self, note_features, pot_arcs):
        z = self.encoder(note_features)
        return self.decoder(z, pot_arcs)


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
        biaffine = True
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
            biaffine
        ).double()
        pos_weight = 1 if pos_weight is None else pos_weight
        self.train_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))
        self.val_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))
        self.val_f1score = BinaryF1Score()
        self.val_f1score_postp = BinaryF1Score()

    def training_step(self, batch, batch_idx):
        note_seq, truth_arcs_mask, pot_arcs = batch
        note_seq, truth_arcs_mask, pot_arcs = note_seq[0], truth_arcs_mask[0], pot_arcs[0]
        num_notes = len(note_seq)
        arc_pred_mask_logits = self.module(note_seq, pot_arcs)
        loss = self.train_loss(arc_pred_mask_logits.float(), truth_arcs_mask.float()).cpu()
        # get predicted class for the edges (e.g. 0 or 1)
        # arc_pred__mask_normalized = torch.sigmoid(arc_pred_mask_logits)
        # arc_pred_mask_bool = torch.round(arc_pred__mask_normalized).bool()
        loss = loss 
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
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
        # compute loss and F1 score
        adj_target = pyg.utils.to_dense_adj(pot_arcs[truth_arcs_mask].T, max_num_nodes=num_notes).squeeze().long().cpu()
        loss = self.val_loss(arc_pred_mask_logits.float(), truth_arcs_mask.float())
        val_fscore = self.val_f1score.cpu()(adj_pred.flatten(), adj_target.flatten())
        self.log("val_loss", loss.item(), batch_size=1)
        self.log("val_fscore", val_fscore.item(), prog_bar=True, batch_size=1)
        # postprocess the values
        ###################################### OLD POSTPROCESSING with only argmax ######################################
        # adj_pred_prob= torch.sparse_coo_tensor(pot_arcs.T, arc_pred__mask_normalized, (num_notes, num_notes)).to_dense().cpu()
        # max_mask = adj_pred_prob.max(dim=0,keepdim=True)[0] == adj_pred_prob
        # adj_pred_postp = adj_pred_prob * max_mask
        # adj_pred_postp[adj_pred_postp!= 0 ] = 1
        # val_fscore_postp = self.val_f1score_postp.cpu()(adj_pred_postp.flatten(), adj_target.flatten())
        # self.log("val_fscore_postp", val_fscore_postp.item(), prog_bar=True, batch_size=1)
        ###################################### NEW POSTPROCESSING with chuliu edmonds ######################################
        adj_pred_logits= torch.sparse_coo_tensor(pot_arcs.T, arc_pred__mask_normalized, (num_notes, num_notes)).cpu().to_dense()
        adj_pred_log_probs = F.logsigmoid(adj_pred_logits)
        head_seq = chuliu_edmonds_one_root(adj_pred_log_probs.numpy().T) # remove the head
        adj_pred_postp = torch.zeros((num_notes,num_notes))
        # dependencies = []
        # for word in self.words:
        #     if word.head == 0:
        #         # make a word for the ROOT
        #         word_entry = {ID: 0, TEXT: "ROOT"}
        #         head = Word(word_entry)
        #     else:
        #         # id is index in words list + 1
        #         head = self.words[word.head - 1]
        #         if word.head != head.id:
        #             raise ValueError("Dependency tree is incorrectly constructed")
        #     self.dependencies.append((head, word.deprel, word))
        for i, head in enumerate(head_seq):
            # TODO: handle the head == 0 case
            if head != 0:
                adj_pred_postp[head, i] = 1
        val_fscore_postp = self.val_f1score_postp.cpu()(adj_pred_postp.flatten(), adj_target.flatten())
        self.log("val_fscore_postp", val_fscore_postp.item(), prog_bar=True, batch_size=1)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, verbose=True)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    