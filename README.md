# ner_service
This project is to train a spacy NER model via API and run NER using the trained model.

### Build a docker image
```bash
$docker build -t ner-service:latest .
```

### Add docker-compose context for ner-service
```dockerfile
ner-service:
    image: ner-service:latest
    container_name: custom-ner
    restart: always
    ports:
      - "9501:9501"
    volumes:
      - ./ner-training:/app/project-ner/training
      - ./ner-training:/app/project-ner/corpus
      - ./ner-training:/app/project-ner/metrics
```

### or Run the docker image
```bash
$docker run -it -p 9501:9501 ner-service:latest
```
