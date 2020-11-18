# ner_service
This project is to train a spacy NER model via API and run NER using the trained model.

### Build a docker image
```bash
$docker build -t ner-service:latest .
```

### Run the docker image
```bash
$docker run -it -p 9501:9501 ner-service:latest
```
