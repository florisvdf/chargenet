#!/usr/bin/env bash

mkdir tmp
unzip -o software/APBS-3.4.1.Linux.zip -d tmp
mv tmp/APBS-3.4.1.Linux/bin/apbs /usr/local/bin/apbs
