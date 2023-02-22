import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torch_geometric as pyg
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import numpy as np

from musicparser.postprocessing import chuliu_edmonds_one_root
from musicparser.data_loading import DURATIONS, get_feats_one_hot


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
        encoder_type = "rnn"
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
            encoder_type
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



class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        encoder_depth,
        n_heads = 4,
        dropout=None,
        activation = "relu"
    ):
        super().__init__()

        if dropout is None:
            dropout = 0

        self.positional_encoder = PositionalEncoding(
            hidden_dim=hidden_dim, dropout=dropout, max_len=200
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, dim_feedforward=hidden_dim, nhead=n_heads, dropout =dropout, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

    def forward(self, z, sentences_len=None):
        z = self.positional_encoder(z)
        z = self.transformer_encoder(z)
        return z, ""


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len, dropout):
        super().__init__()
        
        # Info
        self.dropout = nn.Dropout(dropout)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, hidden_dim)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-np.log(10000.0)) / hidden_dim) # 1000^(2i/hidden_dim)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/hidden_dim))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/hidden_dim))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0)])

class NotesEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        rnn_depth,
        dropout=0,
        embedding_dim = {},
        use_embeddings = True,
        encoder_type = "rnn",
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

        # Encoder layer
        if encoder_type == "rnn":
            self.encoder_cell = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
                bidirectional=bidirectional,
                num_layers=rnn_depth,
                dropout=dropout,
            )
        elif encoder_type == "transformer":
            self.encoder_cell = TransformerEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                encoder_depth=rnn_depth,
                dropout=dropout
            )
        else:
            raise ValueError(f"Encoder type {encoder_type} not supported")
        # embedding layer
        if use_embeddings:
            self.embedding_dim = 128 + len(DURATIONS)+ 1
            self.embedding_pitch = nn.Embedding(128, embedding_dim["pitch"])
            self.embedding_duration = nn.Embedding(len(DURATIONS), embedding_dim["duration"])
            self.embedding_metrical = nn.Embedding(6, embedding_dim["metrical"])
        self.first_linear = nn.Linear(input_dim, hidden_dim)

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
            # concatenate embeddings, we are discarding rests information because it is in the pitch
            z = torch.hstack((pitch, duration, metrical))
        else:
            # one hot encoding
            z = get_feats_one_hot(sequence).double()
        z = self.first_linear(z)
        z, _ = self.encoder_cell(z)
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
            self.bilinear = nn.Bilinear(hidden_channels +1 , hidden_channels + 1, 1)
        else:
            self.lin1 = nn.Linear(2*hidden_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, pot_arcs):
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
            # pass through a dropout layer, shape (num_pot_arcs, hidden_channels)
            input1 = self.dropout(input1)
            input2 = self.dropout(input2)
            # concatenate, like it is done in the stanza parser
            input1 =  torch.cat((input1, torch.ones((input1.shape[0],1), device = input1.device)), dim = -1)
            input2 = torch.cat((input2, torch.ones((input1.shape[0],1), device = input1.device)), dim = -1)
            z = self.bilinear(input1, input2)
        else:
            # concat the embeddings of the two nodes, shape (num_pot_arcs, 2*hidden_channels)
            z = torch.cat([z[pot_arcs[:, 0]], z[pot_arcs[:, 1]]], dim=-1)
            # pass through a linear layer, shape (num_pot_arcs, hidden_channels)
            z = self.lin1(z)
            # pass through activation, shape (num_pot_arcs, hidden_channels)
            z = self.activation(z)
            # pass through another linear layer, shape (num_pot_arcs, 1)
            z = self.lin2(z)
        # return a vector of shape (num_pot_arcs,)
        return z.view(-1)


# class ArcDecoder(torch.nn.Module):
#     def __init__(self, hidden_channels, activation=F.relu, dropout=0.3, biaffine=True):
#         super().__init__()
#         self.activation = activation
#         self.biaffine = biaffine
#         if biaffine:
#             self.scorer = DeepBiaffineScorer(hidden_channels, hidden_channels, hidden_channels, 1, activation, dropout, pairwise=False)
#         else:
#             self.lin1 = nn.Linear(2*hidden_channels, hidden_channels)
#             self.lin2 = nn.Linear(hidden_channels, 1)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, z, pot_arcs):
#         if self.biaffine:
#             z = self.scorer(z[pot_arcs[:, 0]].unsqueeze(0), z[pot_arcs[:, 1]].unsqueeze(0))
#         else:
#             # concat the embeddings of the two nodes, shape (num_pot_arcs, 2*hidden_channels)
#             z = torch.cat([z[pot_arcs[:, 0]], z[pot_arcs[:, 1]]], dim=-1)
#             # pass through a linear layer, shape (num_pot_arcs, hidden_channels)
#             z = self.lin1(z)
#             # pass through activation, shape (num_pot_arcs, hidden_channels)
#             z = self.activation(z)
#             # pass through another linear layer, shape (num_pot_arcs, 1)
#             z = self.lin2(z)
#         # return a vector of shape (num_pot_arcs,)
#         return z.view(-1)



class ArcPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation="relu", dropout=0.2, embedding_dim = {}, use_embedding = True, biaffine = False, encoder_type = "rnn"):
        super().__init__()
        if activation == "relu":
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        self.encoder = NotesEncoder(input_dim, hidden_dim, num_layers, dropout, embedding_dim, use_embedding, encoder_type)
        self.decoder = ArcDecoder(hidden_dim, activation=activation, dropout=dropout, biaffine=biaffine)

    def forward(self, note_features, pot_arcs):
        z = self.encoder(note_features)
        return self.decoder(z, pot_arcs)


# PairwiseBilinear taken from https://github.com/stanfordnlp/stanza/blob/b18e6e80fae7cefbfed7e5255c7ba4ef6f1adae5/stanza/models/common/biaffine.py#L5
 
class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)

        # self.W_bilin.weight.data.zero_()
        # self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size())-1).squeeze()
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size())-1).squeeze()
        return self.W_bilin(input1, input2)


class DeepBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, output_size, activation, dropout, pairwise=True):
        super().__init__()
        self.W1 = nn.Linear(input1_size, hidden_size)
        self.W2 = nn.Linear(input2_size, hidden_size)
        self.activation = activation
        if pairwise:
            raise NotImplementedError
        else:
            self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        return self.scorer(self.dropout(self.activation(self.W1(input1))), self.dropout(self.activation(self.W2(input2))))