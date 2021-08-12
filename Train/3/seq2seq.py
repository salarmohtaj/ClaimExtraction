#url = http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/1_torch_seq2seq_intro.html
import os
import math
import time
import spacy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import List
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator


SEED = 2222
random.seed(SEED)
torch.manual_seed(SEED)

# the link below contains explanation of how spacy's tokenization works
# https://spacy.io/usage/spacy-101#annotations-token
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text: str) -> List[str]:
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text: str) -> List[str]:
    return [tok.text for tok in spacy_en.tokenizer(text)]

source = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
target = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)


train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(source, target))
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

# equivalent, albeit more verbiage train_data.examples[0].src
print(train_data[0].src)
print(train_data[0].trg)

source.build_vocab(train_data, min_freq=2)
target.build_vocab(train_data, min_freq=2)
print(f"Unique tokens in source (de) vocabulary: {len(source.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(target.vocab)}")

BATCH_SIZE = 128

# pytorch boilerplate that determines whether a GPU is present or not,
# this determines whether our dataset or model can to moved to a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create batches out of the dataset and sends them to the appropriate device
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)


# pretend that we're iterating over the iterator and print out the print element
test_batch = next(iter(test_iterator))
print(test_batch)

print(test_batch.src)

# adjustable parameters
INPUT_DIM = len(source.vocab)
OUTPUT_DIM = len(target.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5


class Encoder(nn.Module):
    """
    Input :
        - source batch
    Layer :
        source batch -> Embedding -> LSTM
    Output :
        - LSTM hidden state
        - LSTM cell state

    Parmeters
    ---------
    input_dim : int
        Input dimension, should equal to the source vocab size.

    emb_dim : int
        Embedding layer's dimension.

    hid_dim : int
        LSTM Hidden/Cell state's dimension.

    n_layers : int
        Number of LSTM layers.

    dropout : float
        Dropout for the LSTM layer.
    """

    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, src_batch: torch.LongTensor):
        """

        Parameters
        ----------
        src_batch : 2d torch.LongTensor
            Batched tokenized source sentence of shape [sent len, batch size].

        Returns
        -------
        hidden, cell : 3d torch.LongTensor
            Hidden and cell state of the LSTM layer. Each state's shape
            [n layers * n directions, batch size, hidden dim]
        """
        embedded = self.embedding(src_batch)  # [sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs -> [sent len, batch size, hidden dim * n directions]
        return hidden, cell


encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
hidden, cell = encoder(test_batch.src)
print(hidden.shape, cell.shape)


class Decoder(nn.Module):
    """
    Input :
        - first token in the target batch
        - LSTM hidden state from the encoder
        - LSTM cell state from the encoder
    Layer :
        target batch -> Embedding --
                                   |
        encoder hidden state ------|--> LSTM -> Linear
                                   |
        encoder cell state   -------

    Output :
        - prediction
        - LSTM hidden state
        - LSTM cell state

    Parmeters
    ---------
    output : int
        Output dimension, should equal to the target vocab size.

    emb_dim : int
        Embedding layer's dimension.

    hid_dim : int
        LSTM Hidden/Cell state's dimension.

    n_layers : int
        Number of LSTM layers.

    dropout : float
        Dropout for the LSTM layer.
    """

    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, output_dim)

    def forward(self, trg: torch.LongTensor, hidden: torch.FloatTensor, cell: torch.FloatTensor):
        """

        Parameters
        ----------
        trg : 1d torch.LongTensor
            Batched tokenized source sentence of shape [batch size].

        hidden, cell : 3d torch.FloatTensor
            Hidden and cell state of the LSTM layer. Each state's shape
            [n layers * n directions, batch size, hidden dim]

        Returns
        -------
        prediction : 2d torch.LongTensor
            For each token in the batch, the predicted target vobulary.
            Shape [batch size, output dim]

        hidden, cell : 3d torch.FloatTensor
            Hidden and cell state of the LSTM layer. Each state's shape
            [n layers * n directions, batch size, hidden dim]
        """
        # [1, batch size, emb dim], the 1 serves as sent len
        embedded = self.embedding(trg.unsqueeze(0))
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(outputs.squeeze(0))
        return prediction, hidden, cell


decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)

# notice that we are not passing the entire the .trg
prediction, hidden, cell = decoder(test_batch.trg[0], hidden, cell)
print(prediction.shape, hidden.shape, cell.shape)

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            'Hidden dimensions of encoder and decoder must be equal!'
        assert encoder.n_layers == decoder.n_layers, \
            'Encoder and decoder must have equal number of layers!'

    def forward(self, src_batch: torch.LongTensor, trg_batch: torch.LongTensor,
                teacher_forcing_ratio: float=0.5):

        max_len, batch_size = trg_batch.shape
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder's output
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden & cell state of the encoder is used as the decoder's initial hidden state
        hidden, cell = self.encoder(src_batch)

        trg = trg_batch[0]
        for i in range(1, max_len):
            prediction, hidden, cell = self.decoder(trg, hidden, cell)
            outputs[i] = prediction

            if random.random() < teacher_forcing_ratio:
                trg = trg_batch[i]
            else:
                trg = prediction.argmax(1)

        return outputs

# note that this implementation assumes that the size of the hidden layer,
# and the number of layer are the same between the encoder and decoder
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
seq2seq = Seq2Seq(encoder, decoder, device).to(device)
print(seq2seq)
outputs = seq2seq(test_batch.src, test_batch.trg)
print(outputs.shape)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(seq2seq):,} trainable parameters')


optimizer = optim.Adam(seq2seq.parameters())

# ignore the padding index when calculating the loss
PAD_IDX = target.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train(seq2seq, iterator, optimizer, criterion):
    seq2seq.train()

    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        outputs = seq2seq(batch.src, batch.trg)

        # 1. as mentioned in the seq2seq section, we will
        # cut off the first element when performing the evaluation
        # 2. the loss function only works on 2d inputs
        # with 1d targets we need to flatten each of them
        outputs_flatten = outputs[1:].view(-1, outputs.shape[-1])
        trg_flatten = batch.trg[1:].view(-1)
        loss = criterion(outputs_flatten, trg_flatten)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(seq2seq, iterator, criterion):
    seq2seq.eval()

    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            # turn off teacher forcing
            outputs = seq2seq(batch.src, batch.trg, teacher_forcing_ratio=0)

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            outputs_flatten = outputs[1:].view(-1, outputs.shape[-1])
            trg_flatten = batch.trg[1:].view(-1)
            loss = criterion(outputs_flatten, trg_flatten)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 20
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(seq2seq, train_iterator, optimizer, criterion)
    valid_loss = evaluate(seq2seq, valid_iterator, criterion)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(seq2seq.state_dict(), 'tut1-model.pt')

    # it's easier to see a change in perplexity between epoch as it's an exponential
    # of the loss, hence the scale of the measure is much bigger
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')