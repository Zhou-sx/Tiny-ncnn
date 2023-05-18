#ifndef NCNN_CONCAT_LAYER_H
#define NCNN_CONCAT_LAYER_H
#include "../layer.h"

namespace tiny_ncnn{

class Concat :public Layer{
public:
    Concat() = default;

    int load_param();
    int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

private:
    // int axis; 只支持channel拼接
};

Layer* Concat_layer_creator(void* /*userdata*/);

}
#endif
