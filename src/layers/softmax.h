#ifndef NCNN_Softmax_LAYER_H
#define NCNN_Softmax_LAYER_H

#include "../layer.h"

namespace tiny_ncnn{

class Softmax : public Layer{
public:
    Softmax() { is_inplace = true;}
    
    int load_param(const ParamDict&);
    int forward_inplace(std::vector<Mat>& bottom_blobs) const;


public:
    int axis;
};


Layer* Softmax_layer_creator(void* = nullptr);
    
}

#endif