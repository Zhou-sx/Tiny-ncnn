#include <iostream>
#include <vector>

#include "conv.h"
#include "mat.h"
#include "paramdict.h"
#include "../tools.h"

using namespace tiny_ncnn;


int main(){
    Mat m({9, 9, 1, 3}, 3, 4);

    Mat m1 = m.channel(0);
    m1.fill(0.0f);

    Mat m2 = m.channel(1);
    m2.fill(1.0f);

    Mat m3 = m.channel(2);
    m3.fill(1.0f);

    Layer* p = Convolution_layer_creator();

    FILE* f = fopen("/mnt/c/Users/lenovo/Desktop/Code/Tiny-ncnn/model/conv.param", "rb");
    const DataReader dr = DataReader(f);

    ParamDict param;
    param.load_param(dr);

    (dynamic_cast<Convolution*>(p))->set_model();
    p->load_param(param);

    Mat m4;
    std::vector<Mat> input = {m};
    std::vector<Mat> output = {m4};
    p->forward(input, output);

    m4 = output[0]; // 回写 否则m2不更新
    std::cout << "m value after Conv: " << std::endl;
    pretty_print(m4.channel(1));

    return 0;
}