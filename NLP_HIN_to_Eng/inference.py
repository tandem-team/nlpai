import os
import time
import sys
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
SELF_FILE_PATH = SELF_FILE_PATH.replace("\\", "/")
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
eng_tokens = [tokenize_en(df['english'].values[i]) for i in range(df["english"].shape[0])]
eng_vocab = get_vocab(eng_tokens)
df["eng_tokens"] = eng_tokens
hin_tokens = [indic_tokenize.trivial_tokenize((df['hindi'].values[i])) for i in range(df["hindi"].shape[0])]
hin_vocab = get_vocab(hin_tokens)
df["hin_tokens"] = hin_tokens

df['eng_token_len'] = df['eng_tokens'].apply(lambda x: len(x))
df['hin_token_len'] = df['hin_tokens'].apply(lambda x: len(x))

MAX_LEN = 15
buffer_size = 2
UNK_WORD = '<unk>'
PAD_WORD = '<pad>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"



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
    maxlen = len(sentence[0])
    # print(maxlen)
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


use_gpu = torch.cuda.is_available()
print("use_gpu:", use_gpu)
print(SELF_FILE_PATH)
print("asdfasdfasdf", SELF_FILE_PATH +  "/trained_models/first_trial_model_best.pth.tar")
model = load_checkpoint(file_path=SELF_FILE_PATH +  "/trained_models/first_trial_model_best.tar")
sentences = [str(np.random.choice(df['hindi'].values))]

test_df = pd.read_csv(os.path.dirname(SELF_FILE_PATH) + "/AssignmentNLP/hindiweek1/hindistatements.csv")
test_df = test_df[:10]
sentences = test_df['hindi'].values
translated_sentences = []
sample_size=len(sentences)
start_time=time.time()
for index, sentence in enumerate(sentences):
    # print(f"Original Sentence: {sentence}")
    translated_sentence = greeedy_decode_sentence(model,sentence, use_gpu=use_gpu)
    for punc in [" .", " !", " ,", " - ", " -", "- ", "( ", " )", " ?", " \"", ""]:
        translated_sentence = translated_sentence.replace(punc, punc.replace(" ", ""))
    # print(f"Translated Sentence: {translated_sentence}")
    translated_sentences.append(translated_sentence)


    i = index + 1
    eta = (time.time() - start_time) * (sample_size - i) / i
    sys.stdout.write('\r')
    sys.stdout.write(
              "[%-40s] %d%% [%d/%d]\t ETA: %d Hrs %d Minutes %d Seconds" % (
                      '=' * int(40 * i / sample_size) + '>',
                      100 * i / sample_size,
                      i, sample_size,
                      int(eta // 3600), (eta % 3600) // 60,
                      (eta % 3600) % 60))
    sys.stdout.flush()

test_df['translated_english'] = translated_sentences
df_save = test_df[['hindi', 'translated_english']]
df_save.to_csv(SELF_FILE_PATH + "/inference_output.csv")

