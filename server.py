import torch
import sys
import numpy as np
import pickle
from flask import Flask, jsonify, request

from hLSTM import hLSTM
from glove import Glove, create_emb_layer
import utils
from text_processing import process_text

app = Flask(__name__)

model = torch.load('./models/exp_5/final_model.pth', map_location=torch.device('cpu'))
model.batch_size = 1
model.eval()

MAX_POSTS = 20
MAX_WORDS = 400

glove = Glove()
glove.words = pickle.load(open('./models/exp_5/words.pkl', 'rb'))
glove.word2idx = {o:i for i, o in enumerate(glove.words)}
glove.idx2words = {i:o for i, o in enumerate(glove.words)}

POST_PADDING = [glove.word2idx[glove.pad_token]] * MAX_WORDS # index of pad token is 1

def preprocess(posts):
    if len(posts) > MAX_POSTS:
        posts = posts[-MAX_POSTS:]
    
    posts = [(lambda x: process_text(x))(post) for post in posts]

    for i, post in enumerate(posts):
        if len(post) > MAX_WORDS:
            posts[i] = posts[i][-MAX_WORDS:]

    indices = []
    for post in posts:
        indices = glove.sentence_to_indices(post, seq_len=MAX_WORDS)

    if len(indices) < MAX_POSTS:
        indices.extend([POST_PADDING] * (MAX_POSTS - len(indices)))

    return indices


def get_prediction(posts):
    inputs = preprocess(posts)
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).type('torch.FloatTensor')
    inputs = torch.reshape(inputs, (model.batch_size, MAX_POSTS, -1))
    print(inputs.size(), file=sys.stderr)
    outputs = model.forward(inputs)

    return output.squeeze().tolist()
     

@app.route('/predict', methods=['POST'])
def predict():
    if request.method != 'POST':
        return jsonify({'message': 'Not POST request'})

    args = request.get_json()
    posts = args['posts']

    response = dict()
    response['request_id'] = args['id']

    prediction = get_prediction(posts)
    if len(prediction) > len(posts):
        prediction = prediction[:len(posts)]
    elif len(prediction) < len(posts):
        prediction = ([-1] * (len(posts) - len(prediction))) + prediction

    response['prediction'] = prediction
    
    return jsonify(response)


@app.route('/')
def check():
    return 'Server is live :)))'


if __name__ == '__main__':
    app.run()
