# https://runnable.com/docker/python/
# https://hub.docker.com/_/python/

FROM python:3.6
#FROM tensorflow/tensorflow:nightly-devel-gpu
WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt

CMD [ "python", "./_start.py" ]