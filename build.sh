#!/bin/bash
# NO_COLOR=1 docker build -t mysoft:latest -f Dockerfile .
NO_COLOR=1 docker build -t mysoft:latest -f Dockerfile --progress=plain .

