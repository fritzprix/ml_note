#!/bin/bash
jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
jupyter nbextension install --py --symlink --sys-prefix ipympl
jupyter contrib nbextension install --system

jupyter serverextension enable --py jupyterlab --sys-prefix
jupyter nbextension enable --py --sys-prefix ipympl
jupyter nbextension enable jupyter-black-master/jupyter-black
jupyter nbextension install --py jupyter_tabnine
jupyter nbextension enable --py jupyter_tabnine
jupyter serverextension enable --py jupyter_tabnine