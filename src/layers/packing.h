#ifndef NCNN_Packing_LAYER_H
#define NCNN_Packing_LAYER_H
#include "../layer.h"

namespace tiny_ncnn{

class Packing : public Layer{
public:
    Packing() = default;

    int load_param(const ParamDict&);
    int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;


private:
    int out_elempack;
};

Layer* Packing_layer_creator(void* = nullptr);

}


#endif