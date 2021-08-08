FROM python:3.7-slim-buster
MAINTAINER Dani El-Ayyass <dayyass@yandex.ru>
WORKDIR /app
COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir text-classification-baseline

CMD ["bash"]
