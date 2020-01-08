import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

import utils
from glove import Glove, create_emb_layer
from wrapper import Post, Thread, to_2d_array
from custom_loss import WeightedBCELoss
from hLSTM import hLSTM

def train_and_eval(thread_ids, posts, labels, max_posts=20, 
                    max_words=400, frac=[0.8, 0.1, 0.1], seed=0,
                    batch_size=9, embedding='glove', max_epoch=500):
    
    # preliminary check
    if len(thread_ids) != len(posts) or \
        len(thread_ids) != len(labels) or \
        len(posts) != len(labels):
        raise Exception('Invalid length of data.')

    if len(frac) != 3 or frac[0]+frac[1]+frac[2] != 1:
        raise Exception('Invalid value of frac.')
    
    if frac[0] <= 0 or frac[1] <= 0 or frac[2] <= 0:
        raise Exception('Invalid value(s) for one or more frac element(s).')

    if embedding not in ['glove']:
        raise Exception('Invalid embedding.')

    # create post objects
    posts_obj = []
    for index in len(thread_ids):
        post = Post(thread_ids[index], posts[index], labels[index])
        posts_obj.append(post)

    # create thread objects
    unique_ids = set(thread_ids)
    threads_obj = []
    for thread_id in thread_ids:
        threads_obj.append(Thread(thread_id))

    # put post objects into thread objects
    for post in posts_obj:
        for thread in threads_obj:
            thread.append(post)

    # filter threads
    posts_to_remove = [post for post in posts_obj if len(post) < max_words]
    ids_to_remove = set([post.id for post in posts_to_remove])
    threads_obj = [thread for thread in threads_obj if len(thread) <= max_posts and thread.id not in ids_to_remove]

    # separate thread_ids into train, test, val
    np.random.seed(seed)
    np.random.shuffle(threads_obj)
    train_thd, test_thd, val_thd = np.split(threads_obj, [int(frac[0] * len(threads_obj)), int(frac[0]+frac[1] * len(threads_obj))])

    # [thread][post]
    train_texts, train_labels = to_2d_array(train_thd)
    test_texts, test_labels = to_2d_array(test_thd)
    val_texts, val_labels = to_2d_array(val_thd)

    # from here on is glove specific implementation (may need to extract to a function)
    glove = Glove()
    glove.create_custom_embedding(train_texts)
    glove.add_to_embedding(['.', '!', '?'])

    for i, thread in enumerate(train_texts):
        for j, post_text in enumerate(thread):
            train_texts[i][j] = glove.sentence_to_indices(post_text, seq_len=max_words)
    for i, thread in enumerate(test_texts):
        for j, post_text in enumerate(thread):
            test_texts[i][j] = glove.sentence_to_indices(post_text, seq_len=max_words)
    for i, thread in enumerate(val_texts):
        for j, post_text in enumerate(thread):
            val_texts[i][j] = glove.sentence_to_indices(post_text, seq_len=max_words)

    # still need to add padding for threads length (aka dummy post)


    train_loader = utils.to_data_loader(train_texts, train_labels, batch_size)
    test_loader = utils.to_data_loader(test_texts, test_labels, batch_size)
    val_loader = utils.to_data_loader(val_texts, val_labels, batch_size) 

    # create model
    model = hLSTM(input_size=glove.emb_dim, 
                  hidden_size=glove.emb_dim, 
                  output_size=glove.emb_dim,
                  batch_size=batch_size,
                  num_layers=2, 
                  bidirectional=True, 
                  embedding = create_emb_layer(glove.weights_matrix), 
                  drop_prob=0.5,
                  max_output=max_posts,
                  device=utils.get_device())

    labels = train_labels.flatten()
    intervention_ratio = len(labels[labels == 1]) / len(labels)
    loss_fn = WeightedBCELoss(zero_weight=intervention_ratio, one_weight=1-intervention_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    __train_model(model, train_loader, max_epoch, loss_fn, optimizer)
    f1, precision, recall = __eval_model(model, test_loader, False)


def __train_model(model, data_loader, max_epoch, loss_fn, optimizer):
    model.zero_grad()
    model.train()

    for epoch in range(max_epoch):
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(utils.get_device()), labels.to(utils.get_device())

            # This line is so that the buffers will not be freed when trying to backward through the graph
            h = tuple([each.data for each in h])

            output = model(inputs) # (self.batch_size * self.max_output, 1)
            loss = loss_fn.loss(output.squeeze(), torch.flatten(labels.float()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()


def __eval_model(model, data_loader, temp=True):
    model.eval()

    preds, truths = [], []

    for inputs, labels in data_loader:
        h = tuple([each.data for each in h])

        inputs, labels = inputs.to(utils.get_device()), labels.to(utils.get_device())

        output, h = model(inputs, h)

        pred = torch.round(output.squeeze())
        preds.append(pred.tolist())
        truths.append(torch.flatten(labels).tolist())

    preds = int(torch.flatten(preds))
    truths = torch.flatten(truths)
    
    if temp:
        model.train()
    
    f1 = f1_score(truths, preds)
    precision = precision_score(truths, preds)
    recall = recall_score(truths, preds)

    return f1, precision, recall
