import pandas as pd
import numpy as np

from train_and_eval import train_and_eval

data = pd.read_csv('data/result.csv', comment='#')
thread_ids = np.array(data.thread_id)
posts = np.array(data.posts)
labels = np.array(data.labels)
labels[labels == 'Instructor'] = '1'
labels[labels == 'Student'] = '0'
labels = labels.astype(np.int)

train_and_eval(thread_ids=thread_ids, posts=posts, labels=labels, 
               max_posts=20, max_words=400, frac=[0.8, 0.1, 0.1], seed=0,
               batch_size=9, embedding='glove', max_epoch=2, verbose=True)
