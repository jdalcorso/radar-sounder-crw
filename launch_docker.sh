#!/bin/bash
# Remove/add/modify volumes with -v

docker build --build-arg="USERID=$(id -u)" \
    --build-arg="GROUPID=$(id -g)" \
    --build-arg="REPO_DIR=$(pwd | sed "s/$USER/jordydalcorso/")" \
    -t $USER/$1:$2 .

docker run -h $1 --name $1_$USER \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -u $(id -u):$(id -g) \
    -v /home/$USER:/home/jordydalcorso \
    -v /media:/media \
    -v /home/jordydalcorso/workspace/datasets:/datasets \
    -v /media/datapart/jordydalcorso:/data \
    -w /home/jordydalcorso \
    -it $USER/$1:$2
