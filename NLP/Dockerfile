#FROM pytorch/pytorch:latest
FROM python:3.8.2

RUN apt-get update \
    && apt-get install tesseract-ocr -y

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get install python3.8-dev

#RUN pip3 install torch
#RUN pip3 install torchvision


RUN apt-get update
RUN apt-get install -y make
RUN pip3 install jupyterlab

#RUN pip3 install matplotlib
#RUN pip3 install matplotlib-inline
#RUN pip3 install transformers
RUN pip3 install bertopic[all]

RUN pip3 install pytesseract
RUN pip3 install Pillow
RUN pip3 install easyocr

CMD ["/bin/bash"]
