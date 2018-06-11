#!/bin/bash

wget https://www.dropbox.com/s/ef27hek4l834k75/model_19.h5?dl=1
wget https://www.dropbox.com/s/46g1bq8yte6yvf2/model_23.h5?dl=1
python3 test.py $1 $2 $3
