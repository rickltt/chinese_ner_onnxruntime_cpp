# OnnxRuntime C++

## Requirements

- [Boost](https://github.com/boostorg/boost/releases)
- [glog](https://github.com/google/glog/releases)
- [OnnxRuntime](https://github.com/microsoft/onnxruntime/releases)

## Run

```shell
rm -rf build
mkdir -p build
cd build
cmake ..
make -j8
cd ..

# run
echo "---Running example----"
./build/TestNER 
```