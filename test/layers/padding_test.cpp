#include <iostream>
#include <vector>

#include "padding.h"
#include "mat.h"
#include "../tools.h"

using namespace tiny_ncnn;

int main(){
    Mat m1({4, 4, 1, 3}, 3, 4);

    Layer* p = Padding_layer_creator();
    
    dynamic_cast<Padding*>(p)->set_param(2, 1, 1, 1, 0, 0, 1.5f);

    std::cout << "m value before Padding: " << std::endl;
    pretty_print(m1);

    Mat m2;
    std::vector<Mat> input = {m1};
    std::vector<Mat> output = {m2};
    p->forward(input, output);
    
    m2 = output[0]; // 回写 否则m2不更新

    std::cout << "m value after Padding: " << std::endl;
    pretty_print(m2.channel(0));
}