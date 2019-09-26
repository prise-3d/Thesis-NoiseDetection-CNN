#! /bin/bash

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need to specify where you want to store data"
    exit 1
fi

path=$1

for link in {"data","results","saved_models","models_infos","models_backup","threshold_map","learned_zones","custom_norm"}; do
    rm ${link}
    ln -s $1/${link} ${link}
done