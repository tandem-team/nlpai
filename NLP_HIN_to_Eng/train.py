import os
import time
import sys
import shutil
import spacy

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torchtext.legacy import data
from sklearn.feature_extraction.text import CountVectorizer
from indicnlp.tokenize import sentence_tokenize, indic_tokenize

import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io

import torch.utils.data as data_utils
from torch import nn
from torch.tensor import Tensor, Optional
import torch.nn.functional as F
from torch.autograd import Variable

from model import MyTransformer


SELF_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
print(os.path.dirname(SELF_FILE_PATH))

def get_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)
    return Vocab(counter, specials=['<pad>', '<unk>', '<bos>', '<eos>', '<blank>'])

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


df = pd.read_csv(os.path.dirname(SELF_FILE_PATH) + "/AssignmentNLP/train/train.csv")

spacy_en = spacy.load('en_core_web_sm')

# sentences = str(np.random.choice(df['hindi'].values))
# print(sentences)
# exit()

# SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)

eng_tokens = [tokenize_en(df['english'].values[i]) for i in range(df["english"].shape[0])]
eng_vocab = get_vocab(eng_tokens)
df["eng_tokens"] = eng_tokens

hin_tokens = [indic_tokenize.trivial_tokenize((df['hindi'].values[i])) for i in range(df["hindi"].shape[0])]
hin_vocab = get_vocab(hin_tokens)
df["hin_tokens"] = hin_tokens

for index, i in enumerate(['<unk>', '<pad>', '<bos>', '<eos>', "<blank>"]):
    print(i, eng_vocab.stoi[i])
    print(i, hin_vocab.stoi[i])
    print(i, eng_vocab.itos[index])
    print(i, hin_vocab.itos[index])
print(eng_vocab.stoi["<pad>"])
print(eng_vocab.stoi[5])
print(eng_vocab.stoi[0])

# exit()


df['eng_token_len'] = df['eng_tokens'].apply(lambda x: len(x))
df['hin_token_len'] = df['hin_tokens'].apply(lambda x: len(x))



MAX_LEN = 20
buffer_size = 2
UNK_WORD = '<unk>'
PAD_WORD = '<pad>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'


# clip outlier values
print(df.shape)
df.drop(df[(df['eng_token_len'] > MAX_LEN) + (df['hin_token_len'] > MAX_LEN)].index, inplace = True)
print(df.shape)


# clip the tokens with large size (most of the sentences will have less token difference gap)
df.drop(df[np.abs(df['hin_token_len'].values - df['eng_token_len'].values) != 0].index, inplace=True)
print(df.shape)


# df['eng_token_vec'] = df['eng_tokens'].apply(lambda x: [eng_vocab[t] for t in x] + [0] * (MAX_LEN+buffer_size - len(x))).squeeze()
df['eng_token_vec'] = df['eng_tokens'].apply(lambda x: [eng_vocab.stoi[BOS_WORD]] + [eng_vocab.stoi[t] for t in x] + [eng_vocab.stoi[PAD_WORD]] * (MAX_LEN+buffer_size - len(x) -2)).squeeze()
print(df['eng_tokens'].values[0])
print(df['eng_token_vec'].values[0])
# exit()
# df['hin_token_vec'] = df['hin_tokens'].apply(lambda x: [hin_vocab[t] for t in x] + [0] * (MAX_LEN+buffer_size - len(x))).squeeze()
df['hin_token_vec'] = df['hin_tokens'].apply(lambda x: [hin_vocab.stoi[t] for t in x] + [hin_vocab.stoi[PAD_WORD]] * (MAX_LEN+buffer_size - len(x) -2)).squeeze()

print(df['hin_tokens'].values[0])
print(df['hin_token_vec'].values[0])

df_train=df.sample(frac=0.8,random_state=200) #random state is a seed value
df_val=df.drop(df_train.index)
print("training_size: (df_train)", df_train.shape)
print("validation_size: (df_val)", df_val.shape)
# exit()


def get_dataloaders(train_df, val_df, batch_size):
    train_src = torch.tensor(df['hin_token_vec'].to_list())
    train_trg = torch.tensor(df['eng_token_vec'].to_list())


    train_tensor = data_utils.TensorDataset(train_src, train_trg) 
    train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)

    val_src = torch.tensor(val_df['hin_token_vec'].to_list())
    val_trg = torch.tensor(val_df['eng_token_vec'].to_list())
    val_tensor = data_utils.TensorDataset(val_src, val_trg) 
    val_loader = data_utils.DataLoader(dataset = val_tensor, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader


source_vocab_length = len(hin_vocab)
target_vocab_length = len(eng_vocab)
# Special Tokens
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"


['<unk>', '<pad>', '<bos>', '<eos>']
UNK_WORD = '<unk>'
PAD_WORD = '<pad>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'


# model = MyTransformer(source_vocab_length=source_vocab_length,target_vocab_length=target_vocab_length)

# optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# model = model.cuda()

BATCH_SIZE = 512
print("BATCH_SIZE:", BATCH_SIZE)
train_loader, val_loader = get_dataloaders(df_train, df_val, batch_size=BATCH_SIZE)

print(train_loader)

# for batch_idx, (src, trg) in enumerate(train_loader):
#     print("asdf")
#     print(src.size(), trg.size())


def greeedy_decode_sentence(model,sentence, use_gpu=False):
    model.eval()
    # sentence = SRC.preprocess(sentence)
    sentence = indic_tokenize.trivial_tokenize(sentence)
    indexed = []
    for tok in sentence:
        if hin_vocab.stoi[tok] != 0 :
            indexed.append(hin_vocab.stoi[tok])
        else:
            indexed.append(0)
    trg_init_tok = eng_vocab.stoi[BOS_WORD]
    if use_gpu:
        sentence = Variable(torch.LongTensor([indexed])).cuda()
        trg = torch.LongTensor([[trg_init_tok]]).cuda()
    else:
        sentence = Variable(torch.LongTensor([indexed]))
        trg = torch.LongTensor([[trg_init_tok]])
    
    
    
    translated_sentence = ""
    maxlen = 25
    for i in range(maxlen):
        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        if use_gpu:
            np_mask = np_mask.cuda()
        else:
            np_mask = np_mask
        pred = model(sentence.transpose(0,1), trg, tgt_mask = np_mask)
        add_word = eng_vocab.itos[pred.argmax(dim=2)[-1]]
        translated_sentence+=" "+add_word
        if add_word==EOS_WORD:
            break
        if use_gpu:
            trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
        else:
            trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]])))
        #print(trg)
    return translated_sentence

def train(train_loader, val_loader, model, optim, num_epochs,use_gpu=True, RUN_ID="test_run", save_epochs=20): 
    if use_gpu:
        model.cuda()
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        valid_loss = 0
        # Train model
        model.train()
        
        sample_size=len(train_loader)
        start_time=time.time()

        for batch_idx, (src, trg) in enumerate(train_loader):
            src = src.cuda() if use_gpu else src
            trg = trg.cuda() if use_gpu else trg
            #change to shape (bs , max_seq_len)
            # src = src.transpose(0,1)
            #change to shape (bs , max_seq_len+1) , Since right shifted
            # trg = trg.transpose(0,1)
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            src_mask = (src == 0)
            # print(src_mask)
            # exit()
            # src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
            # src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
            src_mask = src_mask.cuda() if use_gpu else src_mask
            trg_mask = (trg_input == 0)
            # trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
            trg_mask = trg_mask.cuda() if use_gpu else trg_mask
            size = trg_input.size(1)
            #print(size)
            np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
            np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
            np_mask = np_mask.cuda() if use_gpu else np_mask   
            # Forward, backprop, optimizer
            # print(src_mask)
            # print(src_mask.size())
            # print(src.size())
            # print(trg_mask.size())
            # print(trg_input.size())
            # print(np_mask.size())
            # exit()
            optim.zero_grad()
            # preds = model(src.transpose(0,1), trg_input.transpose(0,1), tgt_mask = np_mask)#, src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
            preds = model(src.transpose(0,1), trg_input.transpose(0,1), tgt_mask = np_mask, src_key_padding_mask = src_mask, tgt_key_padding_mask=trg_mask)
            # preds = model(src, trg_input, tgt_mask = np_mask, src_key_padding_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
            preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
            loss = F.cross_entropy(preds,targets, ignore_index=0,reduction='sum')
            loss.backward()
            optim.step()
            train_loss += loss.item()/src.shape[0]
            # print("train_loss", train_loss)
            # print(train_loss/len(train_loader))
            i = batch_idx + 1
            eta = (time.time() - start_time) * (sample_size - i) / i
            sys.stdout.write('\r')
            sys.stdout.write(
                      "[%-40s] %d%% [%d/%d]\t ETA: %d Hrs %d Minutes %d Seconds training_loss: %f" %(
                              '=' * int(40 * i / sample_size) + '>',
                              100 * i / sample_size,
                              i, sample_size,
                              int(eta // 3600), (eta % 3600) // 60,
                              (eta % 3600) % 60,
                              train_loss/(batch_idx+ 1)))
            sys.stdout.flush()
        
        model.eval()
        with torch.no_grad():
            for i, (src, trg) in enumerate(val_loader):
                # print("len(val_loader): ", len(val_loader), src.shape[0])
                src = src.cuda() if use_gpu else src
                trg = trg.cuda() if use_gpu else trg
                #change to shape (bs , max_seq_len)
                # src = src.transpose(0,1)
                #change to shape (bs , max_seq_len+1) , Since right shifted
                # trg = trg.transpose(0,1)
                trg_input = trg[:, :-1]
                targets = trg[:, 1:].contiguous().view(-1)
                src_mask = (src == 0)
                # print("src_mask", src_mask)
                # exit()
                # src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
                src_mask = src_mask.cuda() if use_gpu else src_mask
                trg_mask = (trg_input == 0)
                # trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
                trg_mask = trg_mask.cuda() if use_gpu else trg_mask
                size = trg_input.size(1)
                #print(size)
                np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
                np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
                np_mask = np_mask.cuda() if use_gpu else np_mask

                # preds = model(src.transpose(0,1), trg_input.transpose(0,1), tgt_mask = np_mask)#, src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
                preds = model(src.transpose(0,1), trg_input.transpose(0,1), tgt_mask = np_mask, src_key_padding_mask = src_mask, tgt_key_padding_mask=trg_mask)
                # preds = model(src.transpose(0,1), trg_input.transpose(0,1), tgt_mask = np_mask, src_key_padding_mask = src_mask, tgt_key_padding_mask=trg_mask)
                preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))         
                loss = F.cross_entropy(preds,targets, ignore_index=0,reduction='sum')
                valid_loss += loss.item()/src.shape[0]
            
        # Log after each epoch
        print(f'''Epoch [{epoch+1}/{num_epochs}] complete. Train Loss: {train_loss/len(train_loader):.3f}. Val Loss: {valid_loss/len(val_loader):.3f}''')
        
        #Save best model till now:
        if epoch % save_epochs == 0:
            best_loss_flag = False
            if valid_loss/len(val_loader)<min(valid_losses,default=1e9):
                best_loss_flag = True
        
            train_losses.append(train_loss/len(train_loader))
            valid_losses.append(valid_loss/len(val_loader))

            print("saving state dict")
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_loss': valid_loss/len(val_loader),
                'optimizer': optim.state_dict(),
                'source_vocab_length': source_vocab_length,
                'target_vocab_length': target_vocab_length
                }, best_loss_flag, folder=SELF_FILE_PATH + '/trained_models', RUN_ID=RUN_ID)

        
        # Check Example after each epoch:
        sentences = [str(np.random.choice(df_val['hindi'].values))]
        for sentence in sentences:
            print(f"Original Sentence: {sentence}")
            print(f"Translated Sentence: {greeedy_decode_sentence(model,sentence, use_gpu=use_gpu)}")
    return train_losses,valid_losses


def save_checkpoint(state, is_best, RUN_ID, folder='./', filename='_checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, RUN_ID + filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, RUN_ID + filename),
                        os.path.join(folder, RUN_ID + '_model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MyTransformer(source_vocab_length=checkpoint['source_vocab_length'], target_vocab_length=checkpoint['target_vocab_length'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


model = load_checkpoint(file_path=SELF_FILE_PATH +  "/trained_models/first_trial_model_best.pth.tar")
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()
print("use_gpu:", use_gpu)
RUN_ID = "first_trial"
train(train_loader, val_loader, model, optim, num_epochs=1000,use_gpu=use_gpu, RUN_ID=RUN_ID, save_epochs=20)


# riginal Sentence: मैं टाइग्रेस को नहीं सिखा सका।
# Translated Sentence:  I could n't even teach Tigress Tigress Tigress . I could . I . " . I . I ? I could . I could
# [========================================>] 100% [46/46]     ETA: 0 Hrs 0 Minutes 0 Seconds training_loss: 0.633660Epoch [99/1000] complete. Train Loss: 0.634. Val Loss: 0.187
# saving state dict
# Original Sentence: लेकिन उसका बाप बहुत बडा रोडा है.
# Translated Sentence:  but her father is a big hurdle . - hurdle . a . " . " . a father . but father . but father
# [========================================>] 100% [46/46]     ETA: 0 Hrs 0 Minutes 0 Seconds training_loss: 0.631387Epoch [100/1000] complete. Train Loss: 0.631. Val Loss: 0.169
# saving state dict
# Original Sentence: सेलिब्रिटी सैंडविच उनकी खासियत थी |
# Translated Sentence:  Celebrity sandwiches was their thing . Celebrity . Celebrity . Celebrity . Celebrity . Celebrity . Celebrity . Celebrity sandwiches . Celebrity sandwiches Celebrity sandwiches
# [========================================>] 100% [46/46]     ETA: 0 Hrs 0 Minutes 0 Seconds training_loss: 0.608656Epoch [101/1000] complete. Train Loss: 0.609. Val Loss: 0.174
# Original Sentence: मुझ से चिंता की कोई बात नहीं मिला है।
# Translated Sentence:  Ai n't got nothing to worry about from me by worry . It . It . It . It 's one . It 's got
# [========================================>] 100% [46/46]     ETA: 0 Hrs 0 Minutes 0 Seconds training_loss: 0.591060Epoch [102/1000] complete. Train Loss: 0.591. Val Loss: 0.166
# saving state dict
# Original Sentence: नंबर तीन : प्रचण्ड बनो।
# Translated Sentence:  Number three : be fierce . be . 12033 . listen . . . . . . target . target . target . target .
# [=============>                          ] 32% [15/46]   ETA: 0 Hrs 0 Minutes 32 Seconds training_loss: 0.529886



# Translated Sentence:  Well , before I make that type of investment , look like that . " . " . Well . Well . Well . Well
# [========================================>] 100% [46/46]     ETA: 0 Hrs 0 Minutes 0 Seconds training_loss: 0.369448Epoch [19/1000] complete. Train Loss: 0.369. Val Loss: 0.093
# Original Sentence: कक्ष 237?
# Translated Sentence:  Room 237 ? ? ? ? Room ? Room ? 237 ? ? 237 ? Room ? Room ? Room ? Room ? Room ?
# [========================================>] 100% [46/46]     ETA: 0 Hrs 0 Minutes 0 Seconds training_loss: 0.359516Epoch [20/1000] complete. Train Loss: 0.360. Val Loss: 0.097
# Original Sentence: खाने जाएं तोह चिंता ना करें
# Translated Sentence:  You turn it way down way . You turn way . " idiot . " . " idiot . " idiot . " idiot .
# [========================================>] 100% [46/46]     ETA: 0 Hrs 0 Minutes 0 Seconds training_loss: 0.344896Epoch [21/1000] complete. Train Loss: 0.345. Val Loss: 0.095
# saving state dict
# Original Sentence: - हाँ, बेबी!
# Translated Sentence:  - Yeah , baby ! - ! - Yeah ! - Yeah ! - Yeah ! - Yeah ! - Yeah ! - Yeah Yeah

# Original Sentence: तेज!
# Translated Sentence:  Tej ! Faster ! ! ! Faster ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
# [========================================>] 100% [97/97]     ETA: 0 Hrs 0 Minutes 0 Seconds training_loss: 0.210941Epoch [90/1000] complete. Train Loss: 0.211. Val Loss: 0.065
# Original Sentence: - कि थूक है?
# Translated Sentence:  - Is that spit spit ? that spit spit spit spit spit spit spit spit spit spit spit spit spit spit spit spit spit spit
# [========================================>] 100% [97/97]     ETA: 0 Hrs 0 