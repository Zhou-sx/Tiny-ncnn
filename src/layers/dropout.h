#ifndef NCNN_Dropout_LAYER_H
#define NCNN_Dropout_LAYER_H

#include "../layer.h"

namespace tiny_ncnn{

class Dropout : public Layer{
public:
    Dropout() { is_inplace = true;}

public:

};


Layer* Dropout_layer_creator(void* = nullptr);

}

#endif