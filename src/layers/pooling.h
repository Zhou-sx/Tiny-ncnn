#ifndef LAYER_POOLING_H
#define LAYER_POOLING_H
#include "../layer.h"


namespace tiny_ncnn{

class Pooling : public Layer{
public:
    Pooling() = default;

    int load_param(const ParamDict&);

    int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered) const;

public:
    /*
        PoolMethod_MAX 0
        PoolMethod_AVE 1
    */
    int pooling_type;    // 0

    int kernel_w;        // 1
    int kernel_h;        // 11
    int stride_w;        // 2
    int stride_h;        // 12
    int pad_left;        // 3
    int pad_right;       // 13
    int pad_top;         // 14
    int pad_bottom;      // 15

    int global_pooling;  // 4

};  

Layer* Pooling_layer_creator(void* = nullptr);

}

#endif