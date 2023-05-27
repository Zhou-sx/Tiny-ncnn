#include <iostream>
#include <vector>

#include "conv.h"
#include "mat.h"
#include "paramdict.h"
#include "../tools.h"

using namespace tiny_ncnn;


int main(){
    int size_in = 32;
    int c_in = 256;

    int kernel_sz = 3;
    int stride = 1;
    int padding = 1;
    int dilation = 1;
    int c_out = 384;
    int param_sz = kernel_sz * kernel_sz * c_in * c_out;

    int kernel_extent = dilation * (kernel_sz - 1) + 1;

    int size_out = (size_in - kernel_extent) / stride + 1;

    int op_one = kernel_sz * kernel_sz * 2 * c_in;         // 输出特征图中的一点需要的浮点计算量
    double gflops = size_out * size_out * c_out * op_one * 1.0e-09;

    Mat a = RandomMat({size_in, size_in, 1, c_in}, 3);     // 输入特征图
    Mat b = RandomMat({param_sz, 1, 1, 1}, 1);             // 卷积核参数
    Mat bias = RandomMat({c_out, 1, 1, 1}, 1);             // bias

    Layer* p = Convolution_layer_creator();
    
    /*
        Convolution 0=384 1=3 2=1 3=1 4=1 5=1 6=884736
    */
    ParamDict param;
    param.set(0, c_out);             // c_out
    param.set(1, kernel_sz);         // kernel size
    param.set(2, dilation);          // dilation
    param.set(3, stride);            // stride
    param.set(4, padding);           // padding
    param.set(5, 1);                 // bias_term
    param.set(6, param_sz);          // weight_data_size

    p->load_param(param);
    static_cast<Convolution*>(p)->weight_data = b;
    static_cast<Convolution*>(p)->bias_data = bias;

    Mat c;
    std::vector<Mat> input = {a};
    std::vector<Mat> output = {c};
    p->forward(input, output);

    c = output[0]; // 回写 否则m2不更新
    std::cout << "m value after Conv: " << std::endl;
    // pretty_print(c.channel(0));
    

    time_test(p, a, gflops);

    return 0;
}