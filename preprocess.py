import pandas as pd
import numpy as np

from PostLabeller import PostLabeller, FlattenerType, InterventionModelType
from text_processing import process_text

print('Reading csv')
user_df = pd.read_csv('data/user.csv', comment='#')
post_df = pd.read_csv('data/post.csv', comment='#')
comment_df = pd.read_csv('data/comment.csv', comment='#')

print('Creating post labeller')
pl = PostLabeller(user_df, post_df, comment_df,
                  FlattenerType.TIMESTAMP, InterventionModelType.GIM)

print('Preprocessing post texts')
posts = pd.Series(pl.posts)
posts = posts.map(lambda x: process_text(x))
posts = posts.replace([''], '<html>')
pl.posts = posts.to_numpy()

def relabel(x):
    if x == 'Instructor':
        return 1
    elif x == 'Student':
        return 0
    else:
        raise Exception('Wut!!!')

print('Relabeling labels')
vrelabel = np.vectorize(relabel)
pl.labels = vrelabel(pl.labels)

print('Saving to csv')
pl.to_csv('data/result_large.csv')
