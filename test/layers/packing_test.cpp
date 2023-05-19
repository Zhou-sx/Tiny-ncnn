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
    return 0;
}