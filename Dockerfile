FROM continuumio/anaconda3:2019.10

RUN apt-get update -y && apt-get install -y cython
RUN mkdir -p /app
COPY requirements.txt /app
WORKDIR /app
RUN conda install --file requirements.txt
CMD jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='ml' &
