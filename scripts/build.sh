if [ -d build ]; then
    rm -rf build/*
else
    mkdir build
fi

cd build
# cmake ..
cmake -DENABLE_CUDA=ON ..
make