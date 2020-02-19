import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from wrapper import Post, Thread, to_2d_list

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_data_loader(batch_size, inputs, labels, masks=None, shuffle=False):
    labels = torch.from_numpy(np.array(labels))
    inputs = torch.from_numpy(np.array(inputs)).type('torch.FloatTensor')

    if masks is None:
        data = TensorDataset(inputs, labels)
    else:
        masks = torch.from_numpy(np.array(masks)).type('torch.ByteTensor')
        data = TensorDataset(inputs, labels, masks)
    
    return DataLoader(data, shuffle=shuffle, batch_size=batch_size, drop_last=True)


def filter_and_shuffle_data(thread_ids, posts, labels, max_words, max_posts, seed, frac):
    print('Creating Thread objects')
    unique_ids = set(thread_ids)
    threads_obj = []
    for thread_id in unique_ids:
        threads_obj.append(Thread(thread_id))

    print('Creating Post objects')
    posts_obj = []
    for index in range(len(thread_ids)):
        # take last (max_words) words from the post text
        post_text = posts[index].strip()
        post_text = post_text.split()[-max_words:]
        post_text = ' '.join(post_text)

        post = Post(thread_ids[index], post_text, labels[index])
        thread = [thread for thread in threads_obj if thread.thread_id == thread_ids[index]]

        assert len(thread) == 1 # coz I'm overly paranoid

        thread[0].append(post)
        posts_obj.append(post)

    # No need filtering anymore!!!
    # print('Filtering threads')
    # posts_to_remove = [post for post in posts_obj if len(post) > max_words]
    # ids_to_remove = set([post.thread_id for post in posts_to_remove])
    # threads_obj = [thread for thread in threads_obj if len(thread) <= max_posts and thread.thread_id not in ids_to_remove]

    for index, thread in enumerate(threads_obj):
        if len(thread) <= max_posts:
            continue
        
        to_split = threads_obj.pop(index)
        posts = thread.posts

        # should still exit
        while(len(posts) > 0):
            if(len(posts) >= max_posts):
                to_append = posts[-max_posts:]
                del posts[-max_posts:]
            else:
                to_append = posts.copy()
                del posts[-len(posts)]
            threads_obj.append(Thread(to_split.thread_id, posts=to_append))

    print('Separating threads into train, test, val')
    np.random.seed(seed)
    np.random.shuffle(threads_obj)
    train_thd, test_thd, val_thd = np.split(threads_obj, [int(frac[0] * len(threads_obj)), int((frac[0]+frac[1]) * len(threads_obj))])
    
    # [thread][post]
    train_texts, train_labels = to_2d_list(train_thd)
    test_texts, test_labels = to_2d_list(test_thd)
    val_texts, val_labels = to_2d_list(val_thd)

    return train_texts, train_labels, test_texts, test_labels, val_texts, val_labels
