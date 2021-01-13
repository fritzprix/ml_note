FROM tensorflow/tensorflow:1.15.4-gpu-py3
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl python3-pip pandoc texlive-xetex texlive-fonts-recommended texlive-generic-recommended
RUN curl -sL https://deb.nodesource.com/setup_15.x | bash -
RUN apt-get install -y nodejs
RUN pip3 install --upgrade pip
RUN pip3 install jupyter jedi==0.17.2
RUN pip3 install pandas scikit-learn numpy matplotlib ipympl autopep8 jupyter_contrib_nbextensions jupyter-tabnine python_language_server investpy
RUN pip3 install "tensorflow-gpu>=1.15,<2.0"
RUN pip3 install jupyterlab Pillow
RUN pip3 install --upgrade tensorflow-hub jupyterthemes pyrtfolio==0.4 
RUN python -m pip install trendet --upgrade
WORKDIR /home/
COPY ./start_server.sh .start_server.sh
COPY ./install_extension.sh .install_extension.sh
RUN ./.install_extension.sh
CMD ./.start_server.sh