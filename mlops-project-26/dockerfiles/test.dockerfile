# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY config.yaml config.yaml
COPY src/ src/
COPY models/ models/
COPY data/processed data/processed
COPY data/raw/train data/raw/train
COPY data/raw/test data/raw/test
COPY data/raw/valid data/raw/valid
COPY run.sh run.sh

RUN chmod a+x run.sh
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

WORKDIR /
CMD ["./run.sh"]