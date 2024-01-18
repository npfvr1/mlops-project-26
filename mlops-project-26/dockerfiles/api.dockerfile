FROM python:3.9
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY /requirements_api.txt /code/requirements_api.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements_api.txt
COPY app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]