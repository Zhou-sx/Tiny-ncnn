#ifndef NCNN_LRN_LAYER_H
#define NCNN_LRN_LAYER_H

#include "../layer.h"

namespace tiny_ncnn{

class LRN : public Layer{
public:
    LRN() { is_inplace = true;}

    int load_param(const ParamDict&);

    int forward_inplace(std::vector<Mat>& bottom_blobs) const;


public:
    /*
        NormRegion_ACROSS_CHANNELS 0 通道间
        NormRegion_WITHIN_CHANNEL  1 通道内
    */
    int region_type;


    int local_size; 
    float alpha;
    float beta;
    float bias;

};


Layer* LRN_layer_creator(void* = nullptr);
    
}

#endif