FROM python:3.8.0-slim as builder

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 

COPY ./app ./opt/app

WORKDIR /opt/app/src


CMD ["python3", "serve.py"]


