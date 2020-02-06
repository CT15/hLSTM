import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score

import utils
from glove import Glove, create_emb_layer
from custom_loss import WeightedBCELoss
from hLSTM_CRF import hLSTM_CRF

def train_and_eval_crf(thread_ids, posts, labels, max_posts=20,
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
    
    train_texts, train_labels, test_texts, test_labels, val_texts, val_labels = utils.filter_and_shuffle_data(thread_ids, posts, labels, max_words, max_posts, seed, frac)
    
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
    
    train_masks, test_masks, val_masks = [], [], []
    def get_to_append(ones):
        to_append = [1] * len(labels)
        if len(to_append) < max_posts:
            to_append.extend([0] * (max_posts-len(labels)))
        return to_append

    for labels in train_labels:
        to_append = get_to_append(len(labels))
        train_masks.append(to_append)

    for labels in test_labels:
        to_append = get_to_append(len(labels))
        test_masks.append(to_append)

    for labels in val_labels:
        to_append = get_to_append(len(labels))
        val_masks.append(to_append)

    for labels in [train_labels, test_labels, val_labels]:
        for sublist in labels:
            if len(sublist) < max_posts:
                sublist.extend([0] * (max_posts-len(sublist)))

    train_loader = utils.to_data_loader(batch_size, train_texts, train_labels, train_masks)
    test_loader = utils.to_data_loader(batch_size, test_texts, test_labels, test_masks)
    val_loader = utils.to_data_loader(batch_size, val_texts, val_labels, val_masks)

    print('Creating model')
    embedding = create_emb_layer(torch.from_numpy(glove.weights_matrix).float().to(utils.get_device()))
    model = hLSTM_CRF(num_tags=2,
                      input_size=glove.emb_dim,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    writer = None
    if result_dir is not None:
        writer = SummaryWriter(f'runs/{result_dir}')

        if not os.path.exists(f'models/{result_dir}'):
            os.makedirs(f'models/{result_dir}')

    if not validate:
        val_loader = None

    print('Start training model')
    model.zero_grad()
    model.train()

    running_loss = 0.0
    for epoch in range(max_epoch):
        if (epoch + 1) % 20 == 0:
            print(f'Training model ({epoch + 1} / {max_epoch})')

        for i, (inputs, labels, masks) in enumerate(train_loader):
            inputs, labels, masks = inputs.to(utils.get_device()), labels.to(utils.get_device()), masks.to(utils.get_device())

            loss = model.loss(inputs, labels, masks)

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
                    print('Predictions after training:')

                    with torch.no_grad():
                        scores, seqs = model(x_sent, mask=mask)
                        for score, seq in zip(scores, seqs):
                            str_seq = " ".join(ids_to_tags(seq, ix_to_tag))
                            print('%.2f: %s' % (score.item(), str_seq))


                    f1, _, _ = eval_model(model, val_loader)
                    summary_writer.add_scalar('validation f1', f1,
                                              epoch * len(train_loader) + i)

    print('Evaluating model')
    f1, precision, recall = eval_model(model, test_loader, False)

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
    

def eval_model(model, data_loader, temp=True):
    model.eval()

    preds, truths = [], []

    for inputs, labels, masks in data_loader:
        inputs, labels, masks = inputs.to(utils.get_device()), labels.to(utils.get_device()), masks.to(util.get_device())

        scores, seqs = model(inputs, masks)

        pred = torch.round(seqs.squeeze())
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
