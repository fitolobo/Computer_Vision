FROM python:3.6

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install torch==0.4.1
RUN pip3 install scikit-image==0.14.2
RUN apt-get update
RUN apt-get install -y make
RUN pip3 install jupyterlab

RUN pip3 install sklearn
RUN pip3 install pillow
RUN pip3 install scikit-image
RUN pip3 install ipywidgets --user
RUN pip3 install tqdm
RUN pip3 install pandas
RUN pip3 install opencv-python
RUN pip3 install colormath
RUN pip3 install torchvision

CMD ["/bin/bash"]
