#include <iostream>
#include <vector>

#include "innerproduct.h"
#include "mat.h"
#include "paramdict.h"
#include "../tools.h"

using namespace tiny_ncnn;


int main(){
    Mat m({3, 3, 1, 1}, 2, 4);
    m.fill(1.0f);


    Layer* p = InnerProduct_layer_creator();

    FILE* f = fopen("/mnt/c/Users/lenovo/Desktop/Code/Tiny-ncnn/model/innerproduct.param", "rb");
    const DataReader dr = DataReader(f);

    ParamDict param;
    param.load_param(dr);

    (dynamic_cast<InnerProduct*>(p))->set_model();
    p->load_param(param);

    Mat m4;
    std::vector<Mat> input = {m};
    std::vector<Mat> output = {m4};
    p->forward(input, output);

    m4 = output[0]; // 回写 否则m2不更新
    std::cout << "m value after Conv: " << std::endl;
    pretty_print(m4.channel(0));

    return 0;
}