FROM tensorflow/tensorflow:1.15.4-gpu
RUN apt-get update
RUN apt-get install -y python2.7 curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python2.7 get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
RUN update-alternatives  --set python /usr/bin/python2.7
RUN pip install jupyter
RUN pip install pandas scikit-learn numpy matplotlib
RUN pip install "tensorflow-gpu>=1.15,<2.0"
RUN pip install --upgrade tensorflow-hub jupyterthemes
WORKDIR /home/
COPY ./start_server.sh .start_server.sh
CMD ./.start_server.sh