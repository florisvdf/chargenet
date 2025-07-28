#!/usr/bin/env bash

mkdir tmp
unzip -o THIRD_PARTY_SOFTWARE/pymol-open-source-master.zip -d tmp &&
cd tmp/pymol-open-source-master &&
python setup.py build install --no-launcher
