import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torch_geometric as pyg


class RNNEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        rnn_depth,
        cell_type="GRU",
        dropout=None,
        bidirectional=True,
    ):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("Hidden_dim must be an even integer")
        self.hidden_dim = hidden_dim

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        if dropout is not None:
            self.dropout = dropout
        else:
            self.dropout = 0

        # RNN layer.
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=rnn_depth,
            dropout=self.dropout,
        )

    def forward(self, sentences, sentences_len=None):
        # sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        # rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        # if self.dropout is not None:
        #     rnn_out = self.dropout(rnn_out)
        return rnn_out


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
    def __init__(self, hidden_channels, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z, pot_arcs):
        # concatenate the embeddings of the two elements
        z = torch.cat([z[pot_arcs[:,0]],z[pot_arcs[:,1]]],dim=-1)
        # predict
        z = self.lin1(z)
        z = self.activation(z)
        z = self.lin2(z)
        return z.view(-1)


class ArcPredictionModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, activation=F.relu, dropout=0.2):
		super().__init__()
		self.encoder = RNNEncoder(input_dim, hidden_dim, num_layers)
		self.decoder = ArcDecoder(hidden_dim, activation=activation)

	def forward(self, note_features, pot_arcs):
		z = self.encoder(note_features)
		return self.decoder(z, pot_arcs)

class ArcPredictionLightModel(LightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers=2,
        activation=F.relu,
        dropout=0.3,
        lr=0.001,
        weight_decay=5e-4,
        pos_weight = None,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.module = ArcPredictionModel(
            in_feats,
            n_hidden,
            n_layers,
            activation=activation,
            dropout=dropout,
        ).double()
        pos_weight = 1 if pos_weight is None else pos_weight
        self.train_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))
        self.val_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))

    def training_step(self, batch, batch_idx):
        note_seq, truth_arcs,  truth_arcs_mask, pot_arcs = batch[0]
        num_notes = len(note_seq)
        arc_pred_mask_logits = self.module(note_seq, pot_arcs)
        loss = self.train_loss(arc_pred_mask_logits.float(), truth_arcs_mask.float())
        # get predicted class for the edges (e.g. 0 or 1)
        # arc_pred__mask_normalized = torch.sigmoid(arc_pred_mask_logits)
        # arc_pred_mask_bool = torch.round(arc_pred__mask_normalized).bool()
        loss = loss 
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        note_seq, truth_arcs,  truth_arcs_mask, pot_arcs = batch[0]
        num_notes = len(note_seq)
        arc_pred_mask_logits = self.module(note_seq)
        arc_pred__mask_normalized = torch.sigmoid(arc_pred_mask_logits)
        pred_arc = pot_arcs[:, torch.round(arc_pred__mask_normalized).squeeze().bool()]
        # compute pred and ground truth adj matrices
        if torch.sum(pred_arc) > 0:
            adj_pred = pyg.utils.to_dense_adj(pred_arc, max_num_nodes=num_notes).squeeze().cpu()
        else: # to avoid exception in to_dense_adj when there is no predicted edge
            adj_pred = torch.zeros((num_notes, num_notes)).squeeze().to(self.device).cpu()
        # compute loss and F1 score
        adj_target = pyg.utils.to_dense_adj(truth_arcs, max_num_nodes=num_notes).squeeze().long().cpu()
        loss = self.val_loss(adj_pred.float(), adj_target.float())
        val_fscore = self.val_f1_score.cpu()(adj_pred.flatten(), adj_target.flatten())
        self.log("val_loss", loss.item(), batch_size=1)
        self.log("val_fscore", val_fscore.item(), prog_bar=True, batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, verbose=True)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    