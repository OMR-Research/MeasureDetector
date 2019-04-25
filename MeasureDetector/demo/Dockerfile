FROM tensorflow/tensorflow:1.13.1-py3

RUN pip3 install pillow hug
RUN mkdir -p /usr/src/app

WORKDIR /usr/src/app
COPY inference_server.py .
COPY 2019-04-24_faster-rcnn_inception-resnet-v2.pb .

EXPOSE 8080
CMD ["hug", "-p=8080", "-f=app.py"]