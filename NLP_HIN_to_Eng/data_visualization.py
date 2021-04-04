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

sns.set()

def get_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)
    # print(counter)
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def get_boxplot(df):
    df["sentence"] = np.asarray(df['eng_token_len'])*0
    dd=pd.melt(df,id_vars=["sentence"],value_vars=['eng_token_len','hin_token_len'],var_name='sentence_len')
    sns.boxplot(x='sentence',y='value',data=dd,hue='sentence_len')
    plt.show()    


if __name__=="__main__":
    df = pd.read_csv("../AssignmentNLP/train/train_orig.csv")
    # # df = pd.read_csv("../AssignmentNLP/hindiweek1/hindistatements.csv")
    # print(df.columns)
    # hin_tokens = [indic_tokenize.trivial_tokenize((df['hindi'].values[i])) for i in range(df["hindi"].shape[0])]
    # hin_vocab = get_vocab(hin_tokens)
    # df["hin_tokens"] = hin_tokens
    # df['hin_token_len'] = df['hin_tokens'].apply(lambda x: len(x))
    # ax2 = sns.histplot(df['hin_token_len'].values)
    # plt.show()
    
    # exit()


    spacy_en = spacy.load('en_core_web_sm')

    eng_tokens = [tokenize_en(df['english'].values[i]) for i in range(df["english"].shape[0])]
    eng_vocab = get_vocab(eng_tokens)
    df["eng_tokens"] = eng_tokens

    hin_tokens = [indic_tokenize.trivial_tokenize((df['hindi'].values[i])) for i in range(df["hindi"].shape[0])]
    hin_vocab = get_vocab(hin_tokens)
    df["hin_tokens"] = hin_tokens

    df['eng_token_len'] = df['eng_tokens'].apply(lambda x: len(x))
    df['hin_token_len'] = df['hin_tokens'].apply(lambda x: len(x))
    df["sentence"] = np.asarray(df['hin_token_len'].values)*0
    ax1 = sns.boxplot(df['hin_token_len'].values - df['eng_token_len'].values)
    plt.show()
    print(df.shape)
    df.drop(df[np.abs(df['hin_token_len'].values - df['eng_token_len'].values) !=0].index, inplace=True)
    print(df.shape)

    print(df['hin_token_len'].values - df['eng_token_len'].values)
    ax2 = sns.boxplot(df['hin_token_len'].values - df['eng_token_len'].values)
    plt.show()
    exit()

    print(df.columns)

    dd=pd.melt(df,id_vars=["sentence"],value_vars=['eng_token_len','hin_token_len'],var_name='sentence_len')
    sns.boxplot(x='sentence',y='value',data=dd,hue='sentence_len')
    plt.show()
