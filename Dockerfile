FROM python:3.7-slim as base

RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends \
    # required by psycopg2 at build and runtime
    libpq-dev \
     # required for health check
    curl \
 && apt-get autoremove -y

#FROM base as builder

RUN apt-get update -qq && \
  apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  openssh-client \
  graphviz-dev \
  pkg-config \
  git-core \
  openssl \
  libssl-dev \
  libffi6 \
  libffi-dev \
  libpng-dev

# copy files
COPY . /app/

# change working directory
WORKDIR /app

# install dependencies
RUN pip install -r requirements.txt

# start a new build stage
#FROM base as runner

# change working directory
WORKDIR /app

# change shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# the entry point
#EXPOSE 9501
ENTRYPOINT ["python"]
CMD ["custom_ner_server.py"]
