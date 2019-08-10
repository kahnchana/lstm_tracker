#!/usr/bin/env bash

if [ $# -ne 2 ]
  then
    echo "Model Path and Port number arguments expected"
    exit 1
fi

MODEL_DIR=$1
PORT_NO=$2

cd $MODEL_DIR

nvidia-docker run  \
	    --volume /home:/home \
	    -p $PORT_NO:$PORT_NO \
	    kahnchana/tf:tf1gpu \
        bash -c \
        "cd `pwd`;  \
	tensorboard --port=$PORT_NO --logdir=${MODEL_DIR} &> log_tensorboard.txt"


