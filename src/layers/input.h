#ifndef NCNN_Input_LAYER_H
#define NCNN_Input_LAYER_H

#include "../layer.h"

namespace tiny_ncnn{

class Input : public Layer{
public:
    Input() = default;

    int load_param(const ParamDict&);



public:
    int w; // 1
    int h; // 2
    int d; // 4
    int c; // 3
};


Layer* Input_layer_creator(void* = nullptr);

}

#endif