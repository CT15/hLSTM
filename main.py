import pandas as pd
import numpy as np

from train_and_eval import train_and_eval

data = pd.read_csv('data/partitioned_data/result_large_10.csv', comment='#')

thread_ids = np.array(data.thread_id).astype(str)
posts = np.array(data.posts).astype(str)
labels = np.array(data.labels).astype(int)

train_and_eval(thread_ids=thread_ids, posts=posts, labels=labels, 
               max_posts=100, max_words=70, frac=[0.8, 0.1, 0.1], seed=0,
               batch_size=92, embedding='glove', max_epoch=100,
               validate=True, result_dir='exp_11')
