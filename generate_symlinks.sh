#! /bin/bash


if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need to specify orval you want to use (in /scratch folder)"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No argument supplied"
    echo "Need to specify where you want to store data"
    exit 1
fi


echo "Creating links into /scratch folder"

scratch="/scratch"
orval=$1
path=$2


for link in {"data","results","saved_models","models_info","models_backup","threshold_map","learned_zones","custom_norm"}; do
    
    if [ -L ${link} ]; then
        rm ${link}
    fi
    
    fullpath=${scratch}/${orval}/${path}/${link}

    if [ ! -d "${fullpath}" ]; then
        mkdir -p ${fullpath}
    fi
    
    # remove `orval` name for running part
    ln -s ${scratch}/${path}/${link} ${link}
done