FROM python:3.7-slim-buster
MAINTAINER Dani El-Ayyass <dayyass@yandex.ru>

WORKDIR /workdir

COPY config.yaml ./
COPY data/train.csv data/valid.csv data/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir text-classification-baseline

CMD ["bash"]
