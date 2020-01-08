import numpy as np

class Post(object):
    def __init__(self, thread_id, text, label):
        self.thread_id = thread_id
        self.text = text
        self.label = label

    def __len__(self):
        return len(text.split())


class Thread(object):
    def __init__(self, thread_id, posts=[]):
        self.thread_id = thread_id
        self.posts = posts
    
    def append(self, post):
        if post.thread_id == self.thread_id:
            self.posts.append(post.text)
    
    def __len__(self):
        return len(self.posts)


def to_2d_array(threads):
    posts_arr = np.array([])
    labels_arr = np.array([])

    for thread in threads:
        posts_arr.append(np.array([post.text for post in thread.posts]))
        labels_arr.append(np.array([post.label for post in thread.posts]))
    
    return posts_arr, labels_arr
