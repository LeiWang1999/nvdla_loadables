Compile:

```bash
./nvdla_compiler --prototxt resnet18-cifar10-caffe/deploy.prototxt --caffemodel resnet18-cifar10-caffe/resnet-18.caffemodel --cprecision int8 --calibtable resnet18-cifar10-caffe/resnet18.cifar10.fixed.json --quantizationMode per-kernel
```

**Notice: 经过实验，这里用per-kernel。如果使用per-filter, runtime无论给什么输入输出都是一样的。**

Runtime:

```bash
./nvdla_runtime --loadable fast-math.nvdla --image resnet18-cifar10-caffe/I
mage/175_4.jpg  --rawdump
# Output
Work Found!
Work Done
Shutdown signal received, exiting
Test pass
# cat output.dimg 
0 0 0 0 126 0 0 0 0 0
```

```bash
./nvdla_runtime --loadable cifar10-int8.nvdla --image resnet18-cifar10-caffe/I
mage/cat_32.jpg       --rawdump
# Output
Work Found!
Work Done
Shutdown signal received, exiting
Test pass
# cat output.dimg 
0 0 24 92 8 0 0 0 0 0 
```

