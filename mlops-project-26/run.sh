#!/bin/bash

exec python3 /src/data/make_dataset.py &
exec python3 /src/train_model.py