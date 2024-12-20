import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DoubleLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout, device):
        super(DoubleLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        # Embedding matrices - 20 amino acids + padding
        self.tcr_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        # MLP
        self.hidden_layer = nn.Linear(lstm_dim * 2, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        #padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        #psb
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths.cpu(), batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        #PSB
        lengths = lengths.to(self.device)
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, tcrs, tcr_lens, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        tcr_embeds = self.tcr_embedding(tcrs)
        # LSTM Acceptor
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embeds, tcr_lens)
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1] for i, j in enumerate(tcr_lens)]).view(len(tcr_lens), self.lstm_dim)

        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(peps)
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens), self.lstm_dim)

        # MLP Classifier
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = F.sigmoid(mlp_output)
        return output


class ModifiedDoubleLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout, device, useTrans=False, numHead=4):
        super(ModifiedDoubleLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        self.useT = useTrans
        self.head = numHead

        # Embedding matrices - 20 amino acids + padding
        self.tcr_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)

        if self.useT:
            # Transformer Encoder Layer
            self.transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=32,  # input dimension (embedding dimension)
                nhead=numHead,        # number of attention heads
                dim_feedforward=32*4, # feedforward layer dimension
                dropout=dropout         # dropout rate
            )
            self.transformer_encoder = nn.TransformerEncoder(
                self.transformer_encoder_layer,
                num_layers=2  # number of transformer encoder layers
            )

        # MLP
        """
        self.hidden_layer = nn.Linear(lstm_dim * 2, lstm_dim)
        self.relu = torch.nn.GELU()
        #self.relu = torch.nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        """
        if not self.useT:
            #self.tcr_norm = nn.LayerNorm(lstm_dim)
            #self.pep_norm = nn.LayerNorm(lstm_dim)
            #print(f"Norm Layer: Using 2 Hidden layer {lstm_dim * 8} {lstm_dim * 2} with GELU and dropout {dropout} {dropout / 2}")
            print(f"Using 2 Hidden layer {lstm_dim * 8} {lstm_dim * 2} with GELU and dropout {dropout} {dropout / 2}")
            self.hidden_layer2 = nn.Linear(lstm_dim*2, lstm_dim*8)
            self.relu2 = torch.nn.GELU()
            self.dropout2 = nn.Dropout(p=dropout)
            self.hidden_layer3 = nn.Linear(lstm_dim*8, lstm_dim*2)
            self.relu3 = torch.nn.GELU()
            self.dropout3 = nn.Dropout(p=dropout/2)
            self.output_layer = nn.Linear(lstm_dim * 2, 1)
        else:
            print(f"Using 2 Trans Encoder with head {numHead} and dropout {dropout}")
            self.output_layer = nn.Linear(32, 1)


    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Sort and pack
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths.cpu(), batch_first=True)
        hidden = self.init_hidden(len(lengths))
        lstm_out, hidden = lstm(packed_embeds, hidden)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        _, unperm_idx = perm_idx.sort(0)
        return lstm_out[unperm_idx]

    def forward(self, tcrs, tcr_lens, peps, pep_lens, extract_features=False):
        # TCR Encoder
        tcr_embeds = self.tcr_embedding(tcrs)
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embeds, tcr_lens)
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1] for i, j in enumerate(tcr_lens)]).view(len(tcr_lens), self.lstm_dim)

        # Peptide Encoder
        pep_embeds = self.pep_embedding(peps)
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens), self.lstm_dim)

        # Concatenate features
        if not self.useT:
            #tcr_last_cell = self.tcr_norm(tcr_last_cell)
            #pep_last_cell = self.pep_norm(pep_last_cell)
            tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
        else:
            tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
            tcr_pep_concat = tcr_pep_concat.reshape(-1, 32, 32)
            #tcr_pep_concat = torch.stack((tcr_last_cell, pep_last_cell), dim=1)
            tcr_pep_concat = tcr_pep_concat.permute(1, 0, 2)  # [seq_len, batch_size, lstm_dim*2]

        # Return features if requested
        if extract_features:
            return tcr_pep_concat

        # Pass through MLP
        #hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        if self.useT == False:
            hidden_output = self.dropout2(self.relu2(self.hidden_layer2(tcr_pep_concat)))
            hidden_output = self.dropout3(self.relu3(self.hidden_layer3(hidden_output)))
            mlp_output = self.output_layer(hidden_output)
        else:
            transformer_output = self.transformer_encoder(tcr_pep_concat)  # [seq_len, batch_size, lstm_dim*2]
            hidden_output = transformer_output[-1, :, :]
            mlp_output = self.output_layer(hidden_output)

        output = torch.sigmoid(mlp_output)
        return output



class PaddingAutoencoder(nn.Module):
    def __init__(self, input_len, input_dim, encoding_dim):
        super(PaddingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.encoding_dim = encoding_dim
        # Encoder
        self.encoder = nn.Sequential(
                    nn.Linear(self.input_len * self.input_dim, 300),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(300, 100),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(100, self.encoding_dim))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, self.input_len * self.input_dim))

    def forward(self, batch_size, padded_input):
        concat = padded_input.view(batch_size, self.input_len * self.input_dim)
        encoded = self.encoder(concat)
        decoded = self.decoder(encoded)
        decoding = decoded.view(batch_size, self.input_len, self.input_dim)
        decoding = F.softmax(decoding, dim=2)
        return decoding
    pass


class AutoencoderLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, device, max_len, input_dim, encoding_dim, batch_size, ae_file, train_ae):
        super(AutoencoderLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = encoding_dim
        self.max_len = max_len
        self.input_dim = input_dim
        self.batch_size = batch_size
        # TCR Autoencoder
        self.autoencoder = PaddingAutoencoder(max_len, input_dim, encoding_dim)
        checkpoint = torch.load(ae_file, map_location=device)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        if train_ae is False:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        self.autoencoder.eval()
        # Embedding matrices - 20 amino acids + padding
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.pep_lstm = nn.LSTM(embedding_dim, self.lstm_dim, num_layers=2, batch_first=True, dropout=0.1)
        # MLP
        self.mlp_dim = self.lstm_dim + encoding_dim
        self.hidden_layer = nn.Linear(self.mlp_dim, self.mlp_dim // 2)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(self.mlp_dim // 2, 1)
        self.dropout = nn.Dropout(p=0.1)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, padded_tcrs, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        concat = padded_tcrs.view(self.batch_size, self.max_len * self.input_dim)
        encoded_tcrs = self.autoencoder.encoder(concat)
        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(peps)
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens), self.lstm_dim)
        # MLP Classifier
        tcr_pep_concat = torch.cat([encoded_tcrs, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = F.sigmoid(mlp_output)
        return output
