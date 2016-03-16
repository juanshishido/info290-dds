#!bin/bash

if test -a 'logreg_coefficients.txt'; then
    echo "log_regcoefficients.txt has already been downloaded"
else
    wget https://github.com/dbamman/dds/raw/master/homework/hw3/logreg_coefficients.txt
fi

if test -a 'test_movies.txt'; then
    echo "test_movies.txt has already been downloaded"
else
    wget https://github.com/dbamman/dds/raw/master/homework/hw3/test_movies.txt
fi
