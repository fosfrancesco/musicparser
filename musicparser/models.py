import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torch_geometric as pyg
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy
import numpy as np

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
        data_type = "notes"
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
            data_type
        ).double()
        pos_weight = 1 if pos_weight is None else pos_weight
        self.train_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))
        self.val_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))
        self.val_f1score = BinaryF1Score()
        self.val_f1score_postp = BinaryF1Score()
        self.test_f1score = BinaryF1Score()
        self.test_f1score_postp = BinaryF1Score()

    def training_step(self, batch, batch_idx):
        note_seq, truth_arcs_mask, pot_arcs = batch
        note_seq, truth_arcs_mask, pot_arcs = note_seq[0], truth_arcs_mask[0], pot_arcs[0]
        arc_pred_mask_logits = self.module(note_seq, pot_arcs)
        loss = self.train_loss(arc_pred_mask_logits.float(), truth_arcs_mask.float()).cpu()
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        note_seq, truth_arcs_mask, pot_arcs = batch
        note_seq, truth_arcs_mask, pot_arcs = note_seq[0], truth_arcs_mask[0], pot_arcs[0]
        num_notes = len(note_seq)
        arc_pred_mask_logits = self.module(note_seq, pot_arcs)
        loss = self.val_loss(arc_pred_mask_logits.float(), truth_arcs_mask.float()).cpu()
        self.log("val_loss", loss.item(), batch_size=1)

        arc_pred__mask_normalized = torch.sigmoid(arc_pred_mask_logits)
        pred_arc = pot_arcs[torch.round(arc_pred__mask_normalized).squeeze().bool()]
        # compute pred and ground truth adj matrices
        if torch.sum(pred_arc) > 0:
            adj_pred = pyg.utils.to_dense_adj(pred_arc.T, max_num_nodes=num_notes).squeeze().cpu()
        else: # to avoid exception in to_dense_adj when there is no predicted edge
            adj_pred = torch.zeros((num_notes, num_notes)).squeeze().to(self.device).cpu()
        # compute loss and F1 score
        adj_target = pyg.utils.to_dense_adj(pot_arcs[truth_arcs_mask].T, max_num_nodes=num_notes).squeeze().long().cpu()
        val_fscore = self.val_f1score.cpu()(adj_pred.flatten(), adj_target.flatten())
        self.log("val_fscore", val_fscore.item(), prog_bar=True, batch_size=1)
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
        # compute loss and F1 score
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
        activation = "relu"
    ):
        super().__init__()

        if dropout is None:
            dropout = 0
        self.input_dim = input_dim

        self.positional_encoder = PositionalEncoding(
            d_model=input_dim, dropout=dropout, max_len=200
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, dim_feedforward=hidden_dim, nhead=n_heads, dropout =dropout, activation=activation)
        encoder_norm = nn.LayerNorm(input_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth, norm=encoder_norm)

    def forward(self, z, sentences_len=None):
        # TODO: why this is rescaled like that?
        # z = self.transformer_encoder(z) * np.sqrt(self.input_dim)
        # add positional encoding
        z = self.positional_encoder(z)
        # run transformer encoder
        z = self.transformer_encoder(z)
        return z, ""


# class PositionalEncoding(nn.Module):
#     def __init__(self, pos_enc_dim, max_len, dropout):
#         super().__init__()
        
#         # Info
#         self.dropout = nn.Dropout(dropout)
        
#         # Encoding - From formula
#         pos_encoding = torch.zeros(max_len, pos_enc_dim)
#         positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
#         division_term = torch.exp(torch.arange(0, pos_enc_dim, 2).float() * (-np.log(10000.0)) / pos_enc_dim) # 1000^(2i/hidden_dim)
        
#         # PE(pos, 2i) = sin(pos/1000^(2i/hidden_dim))
#         pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
#         # PE(pos, 2i + 1) = cos(pos/1000^(2i/hidden_dim))
#         pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
#         # Saving buffer (same as parameter without gradients needed)
#         self.register_buffer("pos_encoding",pos_encoding)
        
#     def forward(self, token_embedding: torch.tensor) -> torch.tensor:
#         # Residual connection + pos encoding
#         return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0)])

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
        data_type = "notes"
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
            )
        else:
            raise ValueError(f"Encoder type {encoder_type} not supported")
        # embedding layer
        if use_embeddings:
            if data_type == "notes":
                self.embeddings = nn.ModuleDict({
                    "pitch": nn.Embedding(NUMBER_OF_PITCHES, embedding_dim["pitch"]),
                    "duration": nn.Embedding(len(DURATIONS), embedding_dim["duration"]),
                    "metrical": nn.Embedding(METRICAL_LEVELS, embedding_dim["metrical"])
                })
            elif data_type == "chords":
                # root_numbers, chord_forms, chord_extensions, duration_indices, metrical
                # "root": emb_arg[0], "form": emb_arg[1], "ext": emb_arg[2], "duration": emb_arg[3], "metrical"
                self.embeddings = nn.ModuleDict({
                    "root": nn.Embedding(12, embedding_dim["root"]),
                    "form": nn.Embedding(len(CHORD_FORM), embedding_dim["form"]),
                    "ext": nn.Embedding(len(CHORD_EXTENSION), embedding_dim["ext"]),
                    "duration": nn.Embedding(len(JTB_DURATION), embedding_dim["duration"]),
                    "metrical": nn.Embedding(METRICAL_LEVELS, embedding_dim["metrical"])
                })
            else:
                raise ValueError(f"Data type {data_type} not supported")

    def forward(self, sequence, sentences_len=None):
        if self.use_embeddings:
            # run embedding
            if self.data_type == "notes":
                pitch = sequence[:,0]
                is_rest = sequence[:,1]
                duration = sequence[:,2]
                metrical = sequence[:,3]
                pitch = self.embeddings["pitch"](pitch.long())
                duration = self.embeddings["duration"](duration.long())
                metrical = self.embeddings["metrical"](metrical.long())
                # concatenate embeddings, we are discarding rests information because it is in the pitch
                z = torch.hstack((pitch, duration, metrical))
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
                # concatenate embeddings
                z = torch.hstack((root, form, ext, duration, metrical))
        else:
            # one hot encoding
            z = get_feats_one_hot(sequence).double()
        # z = self.first_linear(z)
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
            # self.bilinear = nn.Bilinear(hidden_channels +1 , hidden_channels + 1, 1)
            self.bilinear = nn.Bilinear(hidden_channels , hidden_channels, 1)
        else:
            self.lin1 = nn.Linear(2*hidden_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_channels)
        

    def forward(self, z, pot_arcs):
        z = self.norm(z)
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
    def __init__(self, input_dim, hidden_dim, num_layers, activation="relu", dropout=0.2, embedding_dim = {}, use_embedding = True, biaffine = False, encoder_type = "rnn", n_heads = 4, data_type = "notes"):
        super().__init__()
        if activation == "relu":
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        self.encoder = NotesEncoder(input_dim, hidden_dim, num_layers, dropout, embedding_dim, use_embedding, encoder_type, activation=activation, n_heads=n_heads, data_type = data_type)
        decoder_dim = hidden_dim if encoder_type == "rnn" else input_dim
        self.decoder = ArcDecoder(decoder_dim, activation=activation, dropout=dropout, biaffine=biaffine)

    def forward(self, note_features, pot_arcs):
        z = self.encoder(note_features)
        return self.decoder(z, pot_arcs)


# # PairwiseBilinear taken from https://github.com/stanfordnlp/stanza/blob/b18e6e80fae7cefbfed7e5255c7ba4ef6f1adae5/stanza/models/common/biaffine.py#L5
 
# class BiaffineScorer(nn.Module):
#     def __init__(self, input1_size, input2_size, output_size):
#         super().__init__()
#         self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)

#         # self.W_bilin.weight.data.zero_()
#         # self.W_bilin.bias.data.zero_()

#     def forward(self, input1, input2):
#         input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size())-1).squeeze()
#         input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size())-1).squeeze()
#         return self.W_bilin(input1, input2)


# class DeepBiaffineScorer(nn.Module):
#     def __init__(self, input1_size, input2_size, hidden_size, output_size, activation, dropout, pairwise=True):
#         super().__init__()
#         self.W1 = nn.Linear(input1_size, hidden_size)
#         self.W2 = nn.Linear(input2_size, hidden_size)
#         self.activation = activation
#         if pairwise:
#             raise NotImplementedError
#         else:
#             self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, input1, input2):
#         return self.scorer(self.dropout(self.activation(self.W1(input1))), self.dropout(self.activation(self.W2(input2))))