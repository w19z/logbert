import sys
sys.path.append('../')

import os
import gc
import pandas as pd
import numpy as np
from logparser import Spell, Drain
import argparse
from tqdm import tqdm
from logdeep.dataset.session import sliding_window

tqdm.pandas()
pd.options.mode.chained_assignment = None

PAD = 0
UNK = 1
START = 2

output_dir = "../output/custom/"
log_file = "comb"


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


if __name__ == "__main__":
    window_size = 2
    step_size = 2
    train_ratio = 0.4

    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')

    # data preprocess
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    
    # for binary data
    #df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    
    df["Label"] = df["Label1"]
    
    df['timestamp'] = df["Datetime"].values.astype(np.int64) // 10 ** 9
    df['deltaT'] = df['Datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)
    # convert time to UTC timestamp
    # df['deltaT'] = df['datetime'].apply(lambda t: (t - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))

    # sampling with fixed window
    # features = ["EventId", "deltaT"]
    # target = "Label"
    # deeplog_df = deeplog_df_transfer(df, features, target, "datetime", window_size=args.w)
    # deeplog_df.dropna(subset=[target], inplace=True)

    # sampling with sliding window
    deeplog_df = sliding_window(df[["timestamp", "Label", "EventId", "deltaT"]],
                                para={
                                    "window_size": int(window_size)*10, 
                                    "step_size": int(step_size) *10}
                                )

    #########
    # Train #
    #########

    df_normal =deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True) #shuffle
    normal_len = len(df_normal)
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    # deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId", "deltaT"])
    deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId"])

    print("training size {}".format(train_len))


    ###############
    # Test Normal #
    ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(os.path.join(output_dir, 'test_normal'), test_normal, ["EventId"])
    print("test normal size {}".format(normal_len - train_len))

    del df_normal
    del train
    del test_normal
    gc.collect()

    #################
    # Test Abnormal #
    #################

    for i in [1, 2, 4]:
        df_abnormal = deeplog_df[deeplog_df["Label"] == i]
        deeplog_file_generator(os.path.join(output_dir,'test_abnormal_{}'.format(i)), df_abnormal, ["EventId"])
        print('test abnormal {} size {}'.format(i, len(df_abnormal)))

