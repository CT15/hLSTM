import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score

import utils
from glove import Glove, create_emb_layer
from wrapper import Post, Thread, to_2d_list
from custom_loss import WeightedBCELoss
from hLSTM import hLSTM

def train_and_eval(thread_ids, posts, labels, max_posts=20, 
                    max_words=400, frac=[0.8, 0.1, 0.1], seed=0,
                    batch_size=9, embedding='glove', max_epoch=500,
                    validate=False, result_dir=None):

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

    print('Filtering threads')
    posts_to_remove = [post for post in posts_obj if len(post) > max_words]
    ids_to_remove = set([post.thread_id for post in posts_to_remove])
    threads_obj = [thread for thread in threads_obj if len(thread) <= max_posts and thread.thread_id not in ids_to_remove]

    print('Separating threads into train, test, val')
    np.random.seed(seed)
    np.random.shuffle(threads_obj)
    train_thd, test_thd, val_thd = np.split(threads_obj, [int(frac[0] * len(threads_obj)), int((frac[0]+frac[1]) * len(threads_obj))])

    # [thread][post]
    train_texts, train_labels = to_2d_list(train_thd)
    test_texts, test_labels = to_2d_list(test_thd)
    val_texts, val_labels = to_2d_list(val_thd)

    # from here on is glove specific implementation (may need to extract to a function)
    print('Init embedding')
    glove = Glove()
    glove.create_custom_embedding([item for sublist in train_texts for item in sublist])
    glove.add_to_embedding(['.', '!', '?'])

    print('Padding and packing data into data loader')
    for i, thread in enumerate(train_texts):
        for j, post_text in enumerate(thread):
            train_texts[i][j] = glove.sentence_to_indices(post_text, seq_len=max_words)
    for i, thread in enumerate(test_texts):
        for j, post_text in enumerate(thread):
            test_texts[i][j] = glove.sentence_to_indices(post_text, seq_len=max_words)
    for i, thread in enumerate(val_texts):
        for j, post_text in enumerate(thread):
            val_texts[i][j] = glove.sentence_to_indices(post_text, seq_len=max_words)

    # padding at the post level
    post_padding = [glove.word2idx[glove.pad_token]] * max_words
    for posts in [train_texts, test_texts, val_texts]:
        for sublist in posts:
            if len(sublist) < max_posts:
                sublist.extend([post_padding] * (max_posts - len(sublist)))
 
    for labels in [train_labels, test_labels, val_labels]:
        for sublist in labels:
            if len(sublist) < max_posts:
                sublist.extend([0] * (max_posts-len(sublist)))

    train_loader = utils.to_data_loader(train_texts, train_labels, batch_size)
    test_loader = utils.to_data_loader(test_texts, test_labels, batch_size)
    val_loader = utils.to_data_loader(val_texts, val_labels, batch_size)

    print('Creating model')
    embedding = create_emb_layer(torch.from_numpy(glove.weights_matrix).float().to(utils.get_device()))
    model = hLSTM(input_size=glove.emb_dim, 
                  hidden_size=glove.emb_dim, 
                  output_size=glove.emb_dim,
                  batch_size=batch_size,
                  num_layers=1, 
                  bidirectional=False,
                  embedding=embedding, 
                  drop_prob=0.5,
                  max_output=max_posts,
                  device=utils.get_device())

    labels = [label for sublist in train_labels for label in sublist]
    
    intervention_ratio = len([label for label in labels if label == 1]) / len(labels)
    loss_fn = WeightedBCELoss(zero_weight=intervention_ratio, one_weight=1-intervention_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    writer = None
    if result_dir is not None:
        writer = SummaryWriter(f'runs/{result_dir}')

        if not os.path.exists(f'models/{result_dir}'):
            os.makedirs(f'models/{result_dir}')

    if not validate:
        val_loader = None

    print('Start training model')
    __train_model(model, train_loader, max_epoch, loss_fn, optimizer, val_loader, writer)

    print('Evaluating model')
    f1, precision, recall = __eval_model(model, test_loader, False)

    print(f'''
    Test results:
    F1 = {f1}
    Precision = {precision}
    Recall = {recall}
    ''')

    if result_dir is not None:
        print('Saving final model')
        torch.save(model.state_dict(), f'models/{result_dir}/final_model.pth')

    print('DONE :)))')


def __train_model(model, train_loader, max_epoch, loss_fn, optimizer, val_loader, summary_writer=None):
    model.zero_grad()
    model.train()

    running_loss = 0.0
    for epoch in range(max_epoch):
        if (epoch + 1) % 20 == 0:
            print(f'Training model ({epoch + 1} / {max_epoch})')

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(utils.get_device()), labels.to(utils.get_device())

            output = model(inputs) # (self.batch_size * self.max_output, 1)
            loss = loss_fn.loss(output.squeeze(), torch.flatten(labels.float()))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999: # every 100 mini-batches
                if summary_writer is not None:
                    summary_writer.add_scalar('training loss', 
                                              running_loss / 1000,
                                              epoch * len(train_loader) + i)
                    running_loss = 0.0
                
                if val_loader is not None:
                    f1, _, _ = __eval_model(model, val_loader)
                    summary_writer.add_scalar('validation f1', f1,
                                              epoch * len(train_loader) + i)


def __eval_model(model, data_loader, temp=True):
    model.eval()

    preds, truths = [], []

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(utils.get_device()), labels.to(utils.get_device())

        output = model(inputs)

        pred = torch.round(output.squeeze())
        preds.append(pred.tolist())
        truths.append(torch.flatten(labels).tolist())

    preds = [int(pred) for predlist in preds for pred in predlist]
    truths = [truth for truthlist in truths for truth in truthlist]
    
    if temp:
        model.train()
    
    f1 = f1_score(truths, preds)
    precision = precision_score(truths, preds)
    recall = recall_score(truths, preds)

    return f1, precision, recall
