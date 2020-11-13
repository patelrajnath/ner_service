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
from spacy.training.iob_utils import biluo_tags_from_offsets

app = Flask(__name__)
api = Api(app)

import flask
flask.__version__


class CustomSpacyNER(Resource):
    def __init__(self, nlp: "Language" = None) -> None:
        self.nlp = nlp
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
        #         X = np.array([args[f] for f in self._required_features]).reshape(1, -1)
        #         y_pred = clf.predict(X)
        #         return {'prediction': y_pred.tolist()[0]}
        print('started sleeping...')
        sleep(10)
        print('sleeping finished...')
        for i in args['text']:
            print('RECEIVED: ', i)
        return {'train': f'received this from you=={args["text"]}'}

    def get(self):
        args = self.reqparse.parse_args()
        #         X = np.array([args[f] for f in self._required_features]).reshape(1, -1)
        #         y_pred = clf.predict(X)
        #         return {'prediction': y_pred.tolist()[0]}
        print('started sleeping...')
        sleep(10)
        print('sleeping finished...')
        for i in args['text']:
            print('RECEIVED: ', i)
        return {'prediction': f'received this from you=={args["text"]}'}


api.add_resource(CustomSpacyNER, '/train', '/predict')

if __name__ == '__main__':
    app.run(debug=False,  port=9501)#host='0.0.0.0',
