FROM continuumio/miniconda3
WORKDIR /data/guandan
# Create the environment:
COPY guandan.yml .
COPY guandan .
RUN conda env create -f guandan.yml
# Activate the environment, and make sure it's activated:
RUN echo "conda activate guandan" >> ~/.bashrc
SHELL ["/bin/bash","--login","-c"]