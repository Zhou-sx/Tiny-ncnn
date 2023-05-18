#ifndef NCNN_LAYER_H
#define NCNN_LAYER_H

#include <string>
#include <vector>
#include <functional>

#include "mat.h"
#include "paramdict.h"
#include "modelbin.h"

namespace tiny_ncnn{


class Layer{
public:
    Layer():type_index(-1) {}
    virtual ~Layer() = default;
    
public:
    virtual int load_param(const ParamDict&);
    virtual int load_model(const ModelBin&);
    int create_pipeline();
    int destroy_pipeline();
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;
    virtual int forward_inplace(std::vector<Mat>& bottom_blobs) const; // bottom_blobs 和 top_blobs 是同一个
    
public:
    // 层类型及层类型的id
    int type_index;
    // std::string type;
    // 层名称和层id
    // int layer_index; 可以不用 由在vector中的序号唯一标识
    std::string name;

    // 该层的输入/输出节点    
    std::vector<int> bottoms;
    std::vector<int> tops;

    // 是否只支持原地操作
    bool is_inplace;

    // 参数通过 Net 和 ParamDict的load_param预存到pd
    // 通过下表索引得参数值 仅在load_param中使用 通过函数参数传值更方便
    // ParamDict pd;
};

}

#endif