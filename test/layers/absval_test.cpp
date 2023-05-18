#include <iostream>
#include <vector>

#include "absval.h"
#include "mat.h"
#include "../tools.h"

using namespace tiny_ncnn;

int main(){
    Mat m({4, 4, 1, 3}, 3, 4);

    Mat m2 = m.channel(0);
    m2.fill(-1.0f);
    std::cout << "m value: " << std::endl;
    pretty_print(m);

    Layer* p = Absval_layer_creator(nullptr);
    
    std::vector<Mat> input = {m};
    p->forward_inplace(input);
    std::cout << "m value after Absval: " << std::endl;
    pretty_print(m);

    
    Mat m3({4, 4, 1, 1}, 3, 4);

    m3.fill(-2.0f);
    std::cout << "m3 value: " << std::endl;
    pretty_print(m3);

    input = {m3};
    p->forward_inplace(input);
    std::cout << "m3 value after Absval: " << std::endl;
    pretty_print(m3);




}