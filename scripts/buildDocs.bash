#!/bin/bash

cd ../docs/source
sphinx-apidoc -o ./ ../../src/ -f -M
cd ..
make html
cd ../scripts
