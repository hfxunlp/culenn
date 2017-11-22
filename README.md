# culenn
CUDA Version Variable Length Extension of torch.nn

There is also a [CPU Version](https://github.com/anoidgit/lenn) of this library to support CPU which you'd better install first.

To install This library, you could just use the following command if you have installed luarocks:  
luarocks install https://raw.githubusercontent.com/anoidgit/culenn/master/rocks/lenn-scm-1.rockspec

## LenSoftMax
The input of this module should be a Lua Table, it consists of 2 elements, the first one is the input matrix of `batchsize*seqlen` to apply softmax function, the second one should be a vector with `batchsize` dimension, the number in each dimension is the actual length to calculate softmax with the corresponding row in input matrix, data beyong that length will be filled with 0. The [test script](https://github.com/anoidgit/culenn/blob/master/test/lensoftmax.lua) can be regarded as an example if you have problem in understanding how to use it.
