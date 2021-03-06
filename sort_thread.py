import pandas as pd

stats = pd.read_csv('./data/stats/stats.csv', index_col=False)
 # order based on total_1
stats = stats.sort_values(by=['total_1'], ascending=False)
descending_ids = list(stats.ids)
# stats = pd.read_csv('./data/stats/sorted_stats.csv', index_col=False)
# descending_ids = list(stats.ids)

data = pd.read_csv('./data/result_large.csv')
data.thread_id = data.thread_id.astype("category")
data.thread_id.cat.set_categories(descending_ids, inplace=True)
data = data.sort_values(['thread_id'])
data.to_csv('./data/trial1.csv', index=False)

print('DONE!')