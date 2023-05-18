#ifndef NCNN_ABSVAL_LAYER_H
#define NCNN_ABSVAL_LAYER_H
#include "../layer.h"

namespace tiny_ncnn{

class Absval :public Layer{
public:
    Absval() { is_inplace = true; };

    int forward_inplace(std::vector<Mat>& bottom_blobs) const;

};

Layer* Absval_layer_creator(void* /*userdata*/);

}
#endif
