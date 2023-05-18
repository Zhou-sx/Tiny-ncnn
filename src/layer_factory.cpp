#include "layer_factory.h"

namespace tiny_ncnn{

std::unordered_map<std::string, int> layer_name_to_index ={
    {"Absval", 0},
    {"Split",  1},
    {"Concat", 2},
    {"Padding",3},
    {"Convolution", 4},
    {"Input", 5},
    {"ReLU",  6},
    {"LRN",   7},
    {"Dropout",8},
    {"Pooling",9},
    {"ConvolutionDepthWise", 10},
    {"InnerProduct", 11},
    {"Softmax", 12}
};

// 工厂模式
std::function<Layer*(void*)> layer_registry[128] ={
    Absval_layer_creator,
    Split_layer_creator,
    Concat_layer_creator,
    Padding_layer_creator,
    Convolution_layer_creator,
    Input_layer_creator,
    ReLU_layer_creator,
    LRN_layer_creator,
    Dropout_layer_creator,
    Pooling_layer_creator,
    ConvolutionDepthWise_layer_creator,
    InnerProduct_layer_creator,
    Softmax_layer_creator

};

int layer_to_index(const std::string& type){
    if(!layer_name_to_index.count(type)){
        return -1;
    }
    return layer_name_to_index[type];
}

Layer*  create_layer(int idx){
    return (layer_registry[idx])(nullptr);
}

Layer* create_layer(const std::string& type){
    int index = layer_to_index(type);
    if (index == -1)
        return 0;
    Layer* layer = create_layer(index);
    layer->type_index = index;
    return layer;
}

}