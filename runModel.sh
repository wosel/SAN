#!/bin/bash

while getopts "h?d:f:" opt; do
    case "$opt" in
    h|\?)
        echo "-d device"
        exit 0
        ;;
    d)  device=$OPTARG
        ;;
    f)  outputFile=$OPTARG
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

TF="device=$device,optimizer=fast_run,lib.cnmem=0.8"

export THEANO_FLAGS=$TF

python model.py daquar.mpi.ini $outputFile

