# Docker builder to play with Tensorflow & scikit-learn 
> for those who are learning ML with old books


## Steps
1. build docker image
    ```shell
    $ ./build.sh
    ```
2. run jupyter notebook
   ```shell
   $ ./run.sh
   ```

## local directory binding
> there are 3 sub-directories mounted to docker for convenience. each of them is intended for..
- work : repository for .ipynb files
- tf_logs : data exported from tensorflow (e.g. models, logs, etc.)
- data : sample data for training of your own

## modules included
* python2
* Tensorflow (1.15 non-GPU)
* pandas 
* numpy 
* matplotlib
* scikit-learn 