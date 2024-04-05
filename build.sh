#!/bin/sh

export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/torch/lib:$LD_LIBRARY_PATH

cargo build $1
