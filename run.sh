#!/bin/bash

array=($(cd exec && ls && cd ..))
echo ${array[@]}

for a in ${array[@]}; do
"exec/$a" >> result/"$a".log
done