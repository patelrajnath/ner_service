import os
from time import sleep

import spacy
from flask import Flask
from flask_restful import Resource, Api, reqparse
from joblib import load
import numpy as np
import sys
import os

# MODEL_DIR = os.environ["MODEL_DIR"]
# MODEL_FILE = os.environ["MODEL_FILE"]
# METADATA_FILE = os.environ["METADATA_FILE"]
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
# METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

# print("Loading model from: {}".format(MODEL_PATH))
# clf = load(MODEL_PATH)
from spacy import Language
from spacy.training.iob_utils import biluo_tags_from_offsets

app = Flask(__name__)
api = Api(app)

import flask
flask.__version__


nlp = None


class CustomSpacyNER(Resource):
    def __init__(self) -> None:
        # self._required_features = ['text', 'entities', 'lang']
        self._required_features = ['text']
        self.reqparse = reqparse.RequestParser()
        for feature in self._required_features:
            self.reqparse.add_argument(
                feature, type=list, required=True, location='json',
                help='No {} provided'.format(feature))
        super(CustomSpacyNER, self).__init__()

    def put(self):
        args = self.reqparse.parse_args()
        for i in args['text']:
            print('RECEIVED: ', i)
        global nlp
        print('before check', nlp)
        if nlp is None:
            print('Loading the model')
            nlp = spacy.blank('en')
        return {'train': f'received this from you=={args["text"]}'}

    def get(self):
        args = self.reqparse.parse_args()
        global nlp
        print('in prediction', nlp)
        for i in args['text']:
            print('RECEIVED: ', i)
        return {'prediction': f'received this from you=={args["text"]}'}


api.add_resource(CustomSpacyNER, '/train', '/predict')

if __name__ == '__main__':
    app.run(debug=False,  port=9501)#host='0.0.0.0',
