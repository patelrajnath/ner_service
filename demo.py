import requests
from typing import Any, Dict, List, Text, Optional, Type

TEXT = 'text'
ENTITIES = "entities"


class DemoCustomNER(object):
    """A custom sentiment analysis component"""
    def __init__(
            self, component_config: Dict[Text, Any] = None,
    ) -> None:
        self.component_config = component_config
        super(DemoCustomNER, self).__init__()

    def train(self, training_data, **kwargs):
        """Load the sentiment polarity labels from the text
           file, retrieve training tokens and after formatting
           data train the classifier."""
        entities = []
        texts = []
        for example in training_data:
            texts.append(example.get(TEXT))
            entities.append(example.get(ENTITIES))

        url = 'http://{0}:{1}/train'.format(self.component_config.get('host'),
                                            self.component_config.get('port'))
        data = {"text": texts, "entities": entities, "params": [self.component_config]}
        print(requests.put(url, json=data).text)

    def process(self, message, **kwargs):
        """Retrieve the tokens of the new message, pass it to the classifier
            and append prediction results to the message class."""
        url = 'http://{0}:{1}/predict'.format(self.component_config.get('host'),
                                              self.component_config.get('port'))
        data = {"text": [message.get(TEXT)]}
        response = requests.get(url, json=data)
        json_response = response.json()

        return json_response['entities'][0]


if __name__ == '__main__':
    train_data = [
        {TEXT: "Uber blew through $1 million a week",
         ENTITIES: [{'start': 0, 'end': 4, 'entity': 'ORG'}]},
        {TEXT: "Android Pay expands to Canada.",
         ENTITIES: [{'start': 0, 'end': 11, 'entity': 'PRODUCT'}, {'start': 23, 'end': 29, 'entity': 'GPE'}]},
        {TEXT: "Spotify steps up Asia expansion",
         ENTITIES: [{'start': 0, 'end': 7, 'entity': "ORG"}, {'start': 17, 'end': 21, 'entity': "LOC"}]},
        {TEXT: "Google Maps launches location sharing",
         ENTITIES: [{'start': 0, 'end': 11, 'entity': "PRODUCT"}]},
        {TEXT: "Google rebrands its business apps",
         ENTITIES: [{'start': 0, 'end': 6, 'entity': "ORG"}]},
        {TEXT: "look what i found on google!",
         ENTITIES: [{'start': 21, 'end': 27, 'entity': "PRODUCT"}]},
        {TEXT: "look what i found on Google Search!",
         ENTITIES: [{'start': 21, 'end': 34, 'entity': "PRODUCT"}]}
    ]

    config_defaults = {
        "host": '127.0.0.1',  # For docker-compose use 'host': 'custom-ner' (container name)
        "port": 9501,
        "arch": 'default',  # options {'default', 'transformer'}, For transformer, when we run it
        # for the first time it will take additional time for downloading the roberta-base model and configurations
        "dropout": 0.1,
        "accumulate_gradient": 1,
        "patience": 100,
        "max_steps": 200,
        "eval_frequency": 100,
    }

    demo = DemoCustomNER(config_defaults)
    TRAIN = True
    if TRAIN:
        print(f'System training...')
        demo.train(train_data)
    text_data = [
        {TEXT: "Uber blew through $1 million a week"},
        {TEXT: "Spotify steps up Asia expansion"}
    ]
    print(f'Running entity extraction...')
    for input_text in text_data:
        print(demo.process(input_text))
