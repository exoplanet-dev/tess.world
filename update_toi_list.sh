#!/bin/bash

mkdir -p src/tess_world/nexsci
wget "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=TOI&select=*" -O src/tess_world/nexsci/toi.csv
