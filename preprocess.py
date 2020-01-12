import pandas as pd
from PostLabeller import PostLabeller, FlattenerType, InterventionModelType

user_df = pd.read_csv('data/user.csv', comment='#')
post_df = pd.read_csv('data/post.csv', comment='#')
comment_df = pd.read_csv('data/comment.csv', comment='#')

pl = PostLabeller(user_df, post_df, comment_df,
                  FlattenerType.TIMESTAMP, InterventionModelType.GIM)

pl.to_csv('data/result_large.csv')