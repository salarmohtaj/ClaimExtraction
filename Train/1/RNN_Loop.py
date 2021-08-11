#URLS https://towardsdatascience.com/a-comprehensive-guide-to-neural-machine-translation-using-seq2sequence-modelling-using-pytorch-41c9b84ba350
# https://colab.research.google.com/github/bala-codes/Natural-Language-Processing-NLP/blob/master/Neural%20Machine%20Translation/1.%20Seq2Seq%20%5BEnc%20%2B%20Dec%5D%20Model%20for%20Neural%20Machine%20Translation%20%28Without%20Attention%20Mechanism%29.ipynb#scrollTo=yvt8AUrWvbA_



##########################################
##Different Fields for claim and content##
##########################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import spacy, random

from torchtext.data.metrics import bleu_score
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torchtext.legacy import data
import string
import nltk
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

from nltk.corpus import stopwords
nltk_words = list(stopwords.words('english'))
nltk_words.extend(list(string.punctuation))



num_epochs = 100
#encoder_embedding_size = 300
encoder_embedding_size = 100
#hidden_size = 1024
#hidden_size = 512
#num_layers = 2
#num_layers = 1
#encoder_dropout = float(0.5)
#decoder_embedding_size = 300
decoder_embedding_size = 100
#hidden_size = 1024
#hidden_size = 512
#num_layers = 2
#num_layers = 1

spacy_english = spacy.load("en_core_web_sm")



content = Field(tokenize="spacy", lower=True, init_token="<sos>", eos_token="<eos>",stop_words=nltk_words)
claim = Field(tokenize="spacy", lower=True, init_token="<sos>", eos_token="<eos>",stop_words=nltk_words)
fields = [(None, None), (None, None), ('claim',claim),('content', content)]

df = pd.read_csv("data/finalDataFrame.csv",sep="\t")
df.sample(frac=1)
df.to_csv("data/finalDataFrame1.csv",sep="\t",index=False)

training_data = TabularDataset(path = "data/finalDataFrame1.csv",format = 'tsv',fields = fields,skip_header = True)
train_data, test_data, valid_data = training_data.split(split_ratio=[0.8, 0.1, 0.1])
content.build_vocab(train_data, max_size=10000, min_freq=1, vectors = "glove.6B.100d")
claim.build_vocab(train_data, max_size=10000, min_freq=1, vectors = "glove.6B.100d")

input_size_encoder = len(content.vocab)
output_size = len(claim.vocab)
input_size_decoder = len(claim.vocab)



def get_length_of_tokens(data):
    src = []
    trg = []
    for item in data.examples:
        src.append(len(vars(item)['content']))
        trg.append(len(vars(item)['claim']))
    return src, trg
src_len, trg_len = get_length_of_tokens(train_data)
print(min(src_len), max(src_len), min(trg_len), max(trg_len))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                      batch_size = BATCH_SIZE,
                                                                      sort_within_batch=True,
                                                                      sort_key=lambda x: len(x.content),
                                                                      device = device)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(EncoderLSTM, self).__init__()

        # Size of the one hot vectors that will be the input to the encoder
        self.input_size = input_size

        # Output size of the word embedding NN
        self.embedding_size = embedding_size

        # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
        self.hidden_size = hidden_size

        # Number of layers in the lstm
        self.num_layers = num_layers

        # Regularization parameter
        self.dropout = nn.Dropout(p)
        self.tag = True

        # Shape --------------------> (5376, 300) [input size, embedding dims]
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

        # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
        self.LSTM = nn.LSTM(self.embedding_size, hidden_size, num_layers, dropout=p)

    # Shape of x (26, 32) [Sequence_length, batch_size]
    def forward(self, x):
        # Shape -----------> (26, 32, 300) [Sequence_length , batch_size , embedding dims]
        embedding = self.dropout(self.embedding(x))

        # Shape --> outputs (26, 32, 1024) [Sequence_length , batch_size , hidden_size]
        # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size]
        outputs, (hidden_state, cell_state) = self.LSTM(embedding)

        return hidden_state, cell_state




class DecoderLSTM(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, output_size):
    super(DecoderLSTM, self).__init__()

    # Size of the one hot vectors that will be the input to the encoder
    self.input_size = input_size

    # Output size of the word embedding NN
    self.embedding_size = embedding_size

    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.hidden_size = hidden_size

    # Number of layers in the lstm
    self.num_layers = num_layers

    # Size of the one hot vectors that will be the output to the encoder (English Vocab Size)
    self.output_size = output_size

    # Regularization parameter
    self.dropout = nn.Dropout(p)
    self.tag = True

    # Shape --------------------> (5376, 300) [input size, embedding dims]
    self.embedding = nn.Embedding(self.input_size, self.embedding_size)

    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
    self.LSTM = nn.LSTM(self.embedding_size, hidden_size, num_layers, dropout = p)

    # Shape -----------> (1024, 4556) [embedding dims, hidden size, num layers]
    self.fc = nn.Linear(self.hidden_size, self.output_size)

  # Shape of x (32) [batch_size]
  def forward(self, x, hidden_state, cell_state):

    # Shape of x (1, 32) [1, batch_size]
    x = x.unsqueeze(0)

    # Shape -----------> (1, 32, 300) [1, batch_size, embedding dims]
    embedding = self.dropout(self.embedding(x))

    # Shape --> outputs (1, 32, 1024) [1, batch_size , hidden_size]
    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size] (passing encoder's hs, cs - context vectors)
    outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state))

    # Shape --> predictions (1, 32, 4556) [ 1, batch_size , output_size]
    predictions = self.fc(outputs)

    # Shape --> predictions (32, 4556) [batch_size , output_size]
    predictions = predictions.squeeze(0)

    return predictions, hidden_state, cell_state





class Seq2Seq(nn.Module):
    def __init__(self, Encoder_LSTM, Decoder_LSTM):
        super(Seq2Seq, self).__init__()

        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM

    def forward(self, source, target, tfr=0.5):
        # Shape - Source : (10, 32) [(Sentence length German + some padding), Number of Sentences]
        batch_size = source.shape[1]

        # Shape - Source : (14, 32) [(Sentence length English + some padding), Number of Sentences]
        target_len = target.shape[0]
        target_vocab_size = len(claim.vocab)

        # Shape --> outputs (14, 32, 5766)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size] (contains encoder's hs, cs - context vectors)
        hidden_state_encoder, cell_state_encoder = self.Encoder_LSTM(source)

        # Shape of x (32 elements)
        x = target[0]  # Trigger token <SOS>

        for i in range(1, target_len):
            # Shape --> output (32, 5766)
            output, hidden_state_decoder, cell_state_decoder = self.Decoder_LSTM(x, hidden_state_encoder,
                                                                                 cell_state_encoder)
            outputs[i] = output
            best_guess = output.argmax(1)  # 0th dimension is batch size, 1st dimension is word embedding
            x = target[
                i] if random.random() < tfr else best_guess  # Either pass the next word correctly from the dataset or use the earlier predicted word

        # Shape --> outputs (14, 32, 5766)
        return outputs



decoder_dropout = float(0.5)
learning_rates = [0.01,0.03,0.001,0.003]
dropout = [0.25,0.5,0.75,0.9]
hidden_sizes = [32,64,128,256,512,1024,2048]
num_layerss = [1,2]

for decoder_dropout in dropout:
    for num_layers in num_layerss:
        for hidden_size in hidden_sizes:
            for learning_rate in learning_rates:
                decoder_dropout = float(decoder_dropout)
                encoder_dropout = decoder_dropout
                encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,
                                           hidden_size, num_layers, encoder_dropout).to(device)
                # print(encoder_lstm)
                # Initialize the pretrained embedding
                pretrained_content_embeddings = content.vocab.vectors
                encoder_lstm.embedding.weight.data.copy_(pretrained_content_embeddings)

                decoder_lstm = DecoderLSTM(input_size_decoder, decoder_embedding_size,
                                           hidden_size, num_layers, decoder_dropout, output_size).to(device)
                print(decoder_lstm)
                # Initialize the pretrained embedding
                pretrained_claim_embeddings = claim.vocab.vectors
                decoder_lstm.embedding.weight.data.copy_(pretrained_claim_embeddings)
                print(learning_rates, hidden_size, num_layers, decoder_dropout)

                f_results = open("resutls-"+str(learning_rates).replace(".","_")+"-"+str(hidden_size)+"-"+str(num_layers)+"-"+str(decoder_dropout).replace(".","_")+".text",'w')

                step = 0

                model = Seq2Seq(encoder_lstm, decoder_lstm).to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                pad_idx = claim.vocab.stoi["<pad>"]
                criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
                pytorch_total_params = sum(p.numel() for p in model.parameters())
                pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                def translate_sentence(model, sentence, content, claim, device, max_length=50):
                    spacy_ger = spacy.load("en_core_web_sm")
                    if type(sentence) == str:
                        tokens = [token.text.lower() for token in spacy_ger(sentence)]
                    else:
                        tokens = [token.lower() for token in sentence]
                    tokens.insert(0, content.init_token)
                    tokens.append(content.eos_token)
                    text_to_indices = [content.vocab.stoi[token] for token in tokens]
                    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

                    # Build encoder hidden, cell state
                    with torch.no_grad():
                        hidden, cell = model.Encoder_LSTM(sentence_tensor)

                    outputs = [claim.vocab.stoi["<sos>"]]

                    for _ in range(max_length):
                        previous_word = torch.LongTensor([outputs[-1]]).to(device)

                        with torch.no_grad():
                            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
                            best_guess = output.argmax(1).item()

                        outputs.append(best_guess)

                        # Model predicts it's the end of the sentence
                        if output.argmax(1).item() == claim.vocab.stoi["<eos>"]:
                            break

                    translated_sentence = [claim.vocab.itos[idx] for idx in outputs]
                    return translated_sentence[1:]

                def bleu(data, model, content, claim, device):
                    targets = []
                    outputs = []

                    for example in data:
                        src = vars(example)["content"]
                        trg = vars(example)["claim"]

                        prediction = translate_sentence(model, src, content, claim, device)
                        prediction = prediction[:-1]  # remove <eos> token
                        print("==================")
                        print(trg)
                        print(prediction)
                        print("==================")
                        targets.append([trg])
                        outputs.append(prediction)

                    return bleu_score(outputs, targets)

                def checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss):
                    print('saving')
                    print()
                    state = {'model': model,'best_loss': best_loss,'epoch': epoch,'rng_state': torch.get_rng_state(), 'optimizer': optimizer.state_dict(),}
                    torch.save(state, 'content/checkpoint-NMT')
                    torch.save(model.state_dict(),'content/checkpoint-NMT-SD')

                epoch_loss = 0.0
                best_loss = float("inf")
                best_epoch = -1
                sentence1 = "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster"
                ts1 = []

                for epoch in range(num_epochs):
                    print("Epoch - {} / {}".format(epoch + 1, num_epochs))
                    model.train()
                    #translated_sentence1 = translate_sentence(model, sentence1, german, english, device, max_length=50)
                    #print(f"Translated example sentence 1: \n {translated_sentence1}")
                    #ts1.append(translated_sentence1)
                    train_loss = 0.0
                    #model.train(True)
                    for batch_idx, batch in enumerate(train_iterator):
                        input = batch.content.to(device)
                        target = batch.claim.to(device)

                        # Pass the input and target for model's forward method
                        output = model(input, target)
                        output = output[1:].reshape(-1, output.shape[2])
                        target = target[1:].reshape(-1)

                        # Clear the accumulating gradients
                        optimizer.zero_grad()

                        # Calculate the loss value for every epoch
                        loss = criterion(output, target)

                        # Calculate the gradients for weights & biases using back-propagation
                        loss.backward()

                        # Clip the gradient value is it exceeds > 1
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                        # Update the weights values using the gradients we calculated using bp
                        optimizer.step()
                        step += 1
                        train_loss += loss.item()

                    model.eval()
                    valid_loss = 0
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(valid_iterator):
                            input = batch.content.to(device)
                            target = batch.claim.to(device)

                            # Pass the input and target for model's forward method
                            output = model(input, target)
                            output = output[1:].reshape(-1, output.shape[2])
                            target = target[1:].reshape(-1)

                            # Clear the accumulating gradients
                            optimizer.zero_grad()

                            # Calculate the loss value for every epoch
                            loss = criterion(output, target)

                            # Update the weights values using the gradients we calculated using bp
                            valid_loss += loss.item()

                    train_loss = train_loss / len(train_iterator)
                    valid_loss = valid_loss / len(valid_iterator)

                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        best_epoch = epoch
                        checkpoint_and_save(model, best_loss, epoch, optimizer, valid_loss)
                    if ((epoch - best_epoch) >= 10):
                        print("no improvement in 10 epochs, break")
                        break
                    f_results.write("Train_Loss\t"+str(train_loss))
                    f_results.write("Validation_Loss\t" + str(valid_loss))
                    #print("Train_Loss - {}".format(train_loss))
                    #print("Validation_Loss - {}".format(valid_loss))
                    #print()
                f_results.close()

#print(epoch_loss / len(train_iterator))
#score = bleu(test_data[1:100], model, content, claim, device)
#print(f"Bleu score {score * 100:.2f}")