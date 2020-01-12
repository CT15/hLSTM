import numpy as np

class Post(object):
    def __init__(self, thread_id, text, label):
        self.thread_id = thread_id
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.text.split())


class Thread(object):
    def __init__(self, thread_id, posts=None):
        self.thread_id = thread_id
        self.posts = [] if posts is None else posts
    
    def append(self, post):
        if post.thread_id != self.thread_id:
            raise Exception(f'Expected thread id {self.thread_id}, get {post.thread_id} instead.')
        self.posts.append(post)
    
    def __len__(self):
        return len(self.posts)


def to_2d_list(threads):
    posts = []
    labels = []

    for thread in threads:
        posts.append([post.text for post in thread.posts])
        labels.append([post.label for post in thread.posts])
    
    return posts, labels
