#include <iostream>

#include "tools.h"
#include "layer.h"
using namespace tiny_ncnn;

int main(){
    Layer layer;

    Mat m({4, 4, 1, 3}, 3, 4);
    Mat m2 = m.channel(0);
    m2.fill(1.0f);

    std::vector<Mat> input = {m};
    std::vector<Mat> output;

    layer.forward(input, output);

    Mat m3 = output[0].channel(0);
    std::cout << "m3 value: " << std::endl;
    pretty_print(m3);

}