#!/bin/bash

while getopts "h?d:o:w:" opt; do
    case "$opt" in
    h|\?)
        echo "-d device -o outputFile -w weightsFile"
        exit 0
        ;;
    d)  device=$OPTARG
        ;;
    o)  outputFile=$OPTARG
		;;
    w)  weightsFile=$OPTARG
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

TF="device=$device,optimizer=fast_run,lib.cnmem=0.2"

export THEANO_FLAGS=$TF

python predict.py daquar.mpi.ini $weightsFile $outputFile

