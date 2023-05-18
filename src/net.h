#ifndef NCNN_NET_H
#define NCNN_NET_H

#include <vector>
#include "blob.h"
#include "layer.h"
#include "extractor.h"
#include "data_reader.h"

namespace tiny_ncnn{
class Extractor;

class Net{
public:
    // 由name取索引
    int find_blob_index_by_name(std::string name) const; 
    int find_layer_index_by_name(std::string name) const; 

    // 读取 .param文件
    int load_param(const DataReader& dr);
    int load_param(const char* path);
    // 读取 .bin文件
    int load_model(const DataReader& dr);
    int load_model(const char* path);

    // for test
    void add_blob(Blob blob){
        blobs.push_back(blob);
    }
    void add_layer(Layer* p_layer){
        layers.push_back(p_layer);
    }

private:
    // 深度学习模型中的节点和层
    std::vector<Blob> blobs;
    std::vector<Layer*> layers;

    // 输入 输出
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;

public:
    // 以上构建了计算图, 但没有数据, 数据在Extractor
    friend class Extractor;
    Extractor create_extractor ();
};

}
#endif