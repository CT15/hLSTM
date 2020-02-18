import pandas as pd
from collections import Counter

from wrapper import Post, Thread

data = pd.read_csv('data/result_small.csv')
thread_ids = data.thread_id
posts = data.posts
labels = data.labels

print('Creating Thread objects')
unique_ids = set(thread_ids)
threads_obj = []
for thread_id in unique_ids:
    threads_obj.append(Thread(thread_id))

print('Creating Post objects')
posts_obj = []
for index in range(len(thread_ids)):
    post = Post(thread_ids[index], posts[index], labels[index])
    thread = [thread for thread in threads_obj if thread.thread_id == thread_ids[index]]

    assert len(thread) == 1 # coz I'm overly paranoid

    thread[0].append(post)
    posts_obj.append(post)

thread_ids = []
number_of_posts = []
total_1 = []
total_0 = []
for thread_obj in threads_obj:
    thread_ids.append(thread_obj.thread_id)
    number_of_posts.append(len(thread_obj))
    posts = thread_obj.posts
    t0, t1 = 0, 0
    for post in posts:
        if post.label == 0:
            t0 += 1
        elif post.label == 1:
            t1 += 1
    total_1.append(t1)
    total_0.append(t0)

df = pd.DataFrame(list(zip(thread_ids, number_of_posts, total_1, total_0)), 
               columns =['ids', 'no_of_posts', 'total_1', 'total_0']) 

df.to_csv('stats.csv')

number_of_posts = Counter(number_of_posts)
total_1 = Counter(total_1)
total_0 = Counter(total_0)

number_of_posts = [(k, v) for k, v in number_of_posts.items()]
total_1 = [(k, v) for k, v in total_1.items()]
total_0 = [(k, v) for k, v in total_0.items()]

number_of_posts = pd.DataFrame(number_of_posts, columns = ['no_of_posts', 'freq']) 
total_1 = pd.DataFrame(total_1, columns = ['total_1_in_a_thread', 'freq'])
total_0 = pd.DataFrame(total_0, columns =['total_0_in_a_thread', 'freq'])

number_of_posts.to_csv('no_of_posts.csv')
total_1.to_csv('total_1.csv')
total_0.to_csv('total_0.csv')

print('DONE!')

# number of posts in a thread
# total number of threads
# total intervened posts
# number of intervened posts in every posts
