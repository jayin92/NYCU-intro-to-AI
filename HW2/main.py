import argparse
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader

from bert import BERT, MovieDataset
from model import Ngram
from preprocess import preprocessing_function

warnings.filterwarnings("ignore")


def prepare_data():
    # do not modify
    df_train = pd.read_csv('data/IMDB_train.csv')
    df_test = pd.read_csv('data/IMDB_test.csv')

    return df_train, df_test


def get_argument():
    # do not modify
    opt = argparse.ArgumentParser()
    opt.add_argument("--model_type",
                        type=str,
                        choices=['ngram', 'BERT'],
                        required=True,
                        help="model type")
    opt.add_argument("--preprocess",
                        type=int,
                        help="whether preprocessing, 0 means no and 1 means yes")
    opt.add_argument("--part",
                        type=int,
                        help="specify the part")

    config = vars(opt.parse_args())
    return config


def first_part(model_type, df_train, df_test, N):
    # load and train model
    if model_type == 'ngram':
        model = Ngram(N)
        model.train(df_train)
    else:
        raise NotImplementedError

    # test performance of model
    perplexity = model.compute_perplexity(df_test)
    print("Perplexity of {}: {}".format(model_type, perplexity))

    return model


def second_part(model_type, df_train, df_test, N):
    # configurations
    second_config = {
        'batch_size': 8,
        'epochs': 1,
        'lr': 2e-5,
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    }

    # load model
    if model_type == 'ngram':
        # train n-gram first
        model = first_part(model_type, df_train, df_test, N)
    elif model_type == 'BERT':
        train_data = MovieDataset(df_train)
        test_data = MovieDataset(df_test)
        train_dataloader = DataLoader(train_data, batch_size=second_config['batch_size'])
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        model = BERT('distilbert-base-uncased', second_config)
    else:
        raise NotImplementedError

    # train model
    if model_type == 'ngram':
        model.train_sentiment(df_train, df_test)
    elif model_type == 'BERT':
        model.train_sentiment(train_dataloader, test_dataloader)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # get argument
    config = get_argument()
    model_type, preprocessed = config['model_type'], config['preprocess']
    N = 2                          # we only use bi-gram in this assignment, but you can try different N

    # read and prepare data
    df_train, df_test = prepare_data()
    label_mapping = {'negative': 0, 'positive': 1}
    df_train['sentiment'] = df_train['sentiment'].map(label_mapping)
    df_test['sentiment'] = df_test['sentiment'].map(label_mapping)

    # Part 0: Implement at least three preprocessing methods in preprocess.py
    if preprocessed:
        df_train['review'] = df_train['review'].apply(preprocessing_function)
        df_test['review'] = df_test['review'].apply(preprocessing_function)

    if config['part'] == 1:
        # Part 1: Implement bi-gram model
        first_part(model_type, df_train, df_test, N)
    elif config['part'] == 2:
        # Part 2: Implement and compare the performance of each model
        second_part(model_type, df_train, df_test, N)
