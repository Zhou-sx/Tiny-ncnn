#ifndef NCNN_Padding_LAYER_H
#define NCNN_Padding_LAYER_H
#include "../layer.h"

namespace tiny_ncnn{

class Padding : public Layer{
public:
    Padding() = default;

    // int load_param();
    int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

    void set_param(int top, int bottom, int left, int right, int front, int behind, float value);

private:
    int top;
    int bottom;
    int left;
    int right;
    int front;
    int behind;

    float value;
};

Layer* Padding_layer_creator(void* = nullptr);

}

#endif