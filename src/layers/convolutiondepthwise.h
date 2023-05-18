#ifndef NCNN_ConvolutionDepthWise_LAYER_H
#define NCNN_ConvolutionDepthWise_LAYER_H
#include "../layer.h"


namespace tiny_ncnn{

class ConvolutionDepthWise : public Layer{
public:
    ConvolutionDepthWise() = default;

    int load_param(const ParamDict&);
    int load_model(const ModelBin&);

    int do_forward(const Mat& bottom_blob, Mat& top_blob) const;
    int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

public:
    int c_out;        // 0
    
    int kernel_w;     // 1
    int kernel_h;     // 11
    
    int dilation_w;   // 2
    int dilation_h;   // 12
    
    int stride_w;     // 3
    int stride_h;     // 13

    int pad_left;     // 4
    int pad_top;      // 14
    int pad_right;    // 15
    int pad_bottom;   // 16
    int pad_value;    // 18

    int bias_term;    // 5
    int weight_data_size; // 6
    int group;        // 7  

    // model
    Mat weight_data;
    Mat bias_data;
};

Layer* ConvolutionDepthWise_layer_creator(void* = nullptr);

}

#endif