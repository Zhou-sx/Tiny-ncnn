#ifndef NCNN_Relu_LAYER_H
#define NCNN_Relu_LAYER_H

#include "../layer.h"

namespace tiny_ncnn{

class ReLU : public Layer{
public:
    ReLU() { is_inplace = true;}

    int forward_inplace(std::vector<Mat>& bottom_blobs) const;


public:
};


Layer* ReLU_layer_creator(void* = nullptr);
    
}

#endif