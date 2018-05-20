#!/bin/bash

for i in $(seq 1 $3)
do
    (cd /home/arthurcgusmao/Projects/xkbc/algorithms/pra/; sbt "runMain edu.cmu.ml.rtw.pra.experiments.ExperimentRunner $1 $2")
done
