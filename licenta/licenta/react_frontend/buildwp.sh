#!/bin/bash

startPath="$(pwd)"

echo $startPath

webpack

cp bundle.js built/react_frontend/
cp index.html built/react_frontend/
cp -r style/ built/react_frontend


cd ../../


python manage.py collectstatic --noinput

cd $startPath