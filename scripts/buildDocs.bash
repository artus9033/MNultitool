#!/bin/bash

cd ../docs/source
sphinx-apidoc -o ./ ../../mnultitool/ -f -M
cd ..
make html
cd ../scripts
