if [ -d build ]; then
    rm -rf build/*
else
    mkdir build
fi

cd build
cmake -DENABLE_CUDA=ON ..
# cmake -DENABLE_CUDA=OFF ..
make