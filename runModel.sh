#!/bin/bash

while getopts "h?d:" opt; do
    case "$opt" in
    h|\?)
        echo "-d device"
        exit 0
        ;;
    d)  device=$OPTARG
        ;;
    esac
done

echo $device

if [[ $device == 'cpu' ]] || [[ $device =~ gpu[0123] ]] ; then
 :
else
	echo "bad or no device";
	exit 1;
fi

TF="device=$device,optimizer=fast_compile"

export THEANO_FLAGS=$TF

python model.py daquar.mpi.ini

