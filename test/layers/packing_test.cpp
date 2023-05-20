#include <iostream>
#include <vector>

#include "packing.h"
#include "mat.h"
#include "../tools.h"

using namespace tiny_ncnn;

int main(){
    Mat m1({4, 4, 1, 4}, 3, 4);

    Mat m1_1 = m1.channel(0);
    m1_1.fill(0.0f);

    Mat m1_2 = m1.channel(1);
    m1_2.fill(1.0f);

    Mat m1_3 = m1.channel(2);
    m1_3.fill(2.0f);

    Mat m1_4 = m1.channel(3);
    m1_4.fill(3.0f);

    Layer* p = Packing_layer_creator();
    
    ParamDict param;
    param.set(0, 4);
    p->load_param(param);

    Mat m2;
    std::vector<Mat> input = {m1};
    std::vector<Mat> output = {m2};
    p->forward(input, output);

    m2 = output[0];

    std::cout << "raw" << std::endl;
    pretty_print(m1);

    std::cout << "pack:" << std::endl;
    pretty_print(m2);


    // 修改 Packing Layer 变成 unpack
    param.set(0, 1);
    p->load_param(param);

    Mat m2_unpack;
    input = {m2};
    output = {m2_unpack};
    p->forward(input, output);
    m2_unpack = output[0];
    std::cout << "unpack:" << std::endl;
    pretty_print(m2_unpack);


    // Mat m3 = m2.reshape({2, 8, 1, 1}, 2);
    // std::cout << "pack reshape:" << std::endl;
    // pretty_print(m3);

    // Mat m4 = m3.clone();
    // std::cout << "pack clone:" << std::endl;
    // pretty_print(m4);


    // Mat m5 = m2.reshape({2, 2, 1, 4}, 3);
    // Mat m6= m5.channel(0);
    // std::cout << "pack channel:" << std::endl;
    // pretty_print(m6);

    return 0;
}