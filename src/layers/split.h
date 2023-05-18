#ifndef NCNN_SPLIT_LAYER_H
#define NCNN_SPLIT_LAYER_H

#include "../layer.h"

namespace tiny_ncnn{

class Split :public Layer{
public:
    Split() = default;

    int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

private:
    // int axis; 只支持channel拼接
};

Layer* Split_layer_creator(void* /*userdata*/);

}
#endif