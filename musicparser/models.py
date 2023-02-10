import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


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

    def forward(self, z):
        indices = torch.arange(len(z))
        cart_prod = torch.cartesian_prod(indices,indices)
        # remove self loops, since we don't predict on these
        cart_prod = cart_prod[cart_prod[:,0]!=cart_prod[:,1]]
        # concatenate the embeddings of the two elements
        z = torch.cat([z[cart_prod[:,0]],z[cart_prod[:,1]]],dim=-1)
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

	def forward(self, note_features):
		z = self.encoder(note_features)
		return self.decoder(z)

class VoiceLinkPredictionLightModelPG(LightningModule):
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
        self.train_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))

    def training_step(self, batch, batch_idx):
        note_seq, truth_arcs,  truth_arcs_mask = batch[0]
        num_notes = len(note_seq)
        arc_pred_mask_logits = self.module(note_seq)
        loss = self.train_loss(arc_pred_mask_logits.float(), truth_arcs_mask.float())
        # get predicted class for the edges (e.g. 0 or 1)
        # arc_pred__mask_normalized = torch.sigmoid(arc_pred_mask_logits)
        # arc_pred_mask_bool = torch.round(arc_pred__mask_normalized).bool()
        loss = loss 
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        num_notes = len(graph.x_dict["note"])
        edge_target = graph["truth_edges"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        self.val_metric_logging_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes, linear_assignment=self.linear_assignment
        )

    def test_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        num_notes = len(graph.x_dict["note"])
        edge_target = graph["truth_edges"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        # log without linear assignment
        self.test_metric_logging_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, verbose=True)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    