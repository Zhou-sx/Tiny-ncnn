#include "layer.h"

namespace tiny_ncnn{

int Layer::load_param(const ParamDict&){
    return 0;
}

int Layer::load_model(const ModelBin&){
    return 0;
}   

int Layer::create_pipeline()
{
    return 0;
}

int Layer::destroy_pipeline()
{
    return 0;
}

int Layer::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const{
    // 具体操作（以拷贝作为示例）
    top_blobs = bottom_blobs;
    for (int i = 0; i < (int)top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blobs[i].clone();
        if (top_blobs[i].empty())
            return -100;
    }
    return forward_inplace(top_blobs);
}

int Layer::forward_inplace(std::vector<Mat>& bottom_blobs) const{
    return 0;
}

}