# How to use: `source setenv.sh`

export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/torch/lib:$LD_LIBRARY_PATH

echo LIBTORCH_USE_PYTORCH=$LIBTORCH_USE_PYTORCH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
