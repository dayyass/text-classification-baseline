FROM python:3.7-slim-buster
MAINTAINER Dani El-Ayyass <dayyass@yandex.ru>
WORKDIR /workdir
COPY config.yaml .
COPY data .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir text-classification-baseline

CMD ["bash"]
