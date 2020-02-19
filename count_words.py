import pandas as pd

data = pd.read_csv('./data/partitioned_data/result_large_10.csv')
texts = data.posts

count = dict()
for text in texts:
    length = len(text.strip().split())
    current_count = count.get(length, 0)
    count[length] = (current_count + 1)

temp = 0
for k, v in count.items():
    if k <= 70:
        print(k, v)
        temp += v

print(temp)
    
