#!/bin/bash

startPath="$(pwd)"

echo startPath

webpack

cp bundle.js built/
cp index.html built/

cd ../../

python2 manage.py collectstatic --noinput

cd startPath