#!bin/bash

FILES="logreg_coefficients.txt test_movies.txt"

for f in $FILES
do
    if test -a $f; then
        echo "$f has already been downloaded"
    else
        wget https://github.com/dbamman/dds/raw/master/homework/hw3/$f
    fi
done
