FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN apt-get update
RUN curl -sL https://deb.nodesource.com/setup_15.x | bash -
RUN apt-get install -y nodejs xvfb python-opengl
RUN pip install jupyter jedi==0.17.2
RUN pip install --upgrade scikit-learn matplotlib ipympl autopep8 jupyter_contrib_nbextensions python_language_server transformers torchvision  d2l
RUN pip install --upgrade jupyterlab Pillow wandb
RUN pip install --upgrade jupyterthemes pandas
WORKDIR /home/
COPY ./start_server.sh .start_server.sh
COPY ./install_extension.sh .install_extension.sh
RUN ./.install_extension.sh
CMD ./.start_server.sh
