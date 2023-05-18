#include "split.h"

namespace tiny_ncnn{

int Split::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const{
    const Mat& bottom_blob = bottom_blobs[0];
    for (size_t i=0; i<top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }
    return 0;
}

Layer* Split_layer_creator(void* /*userdata*/){
    return new(Split);
}

}