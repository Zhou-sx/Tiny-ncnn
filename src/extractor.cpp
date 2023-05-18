#include "extractor.h"
#include "platform.h"

namespace tiny_ncnn{

Extractor::Extractor(Net* const _net): net(_net) {
    int n = net->blobs.size();
    blob_mats.resize(n);
}

int Extractor::input(int index, const Mat& v){
    if(index < 0 || index >= blob_mats.size()){
        NCNN_LOGE("There is no blob index(%d), input set error!", index);
        return -1;
    }
    blob_mats[index] = v;
    return 0;
}

int Extractor::input(string name, const Mat& v){
    int index = net->find_blob_index_by_name(name);
    if(index == -1){
        NCNN_LOGE("There is no blob named %s, input set error!", name.c_str());
    }
    
    return input(index, v);
}

/*
    先检查调用layer_index需要哪些节点
    而这些节点又需要调用哪些层
    所以形成了递归
    记忆化 如果发现节点已经有数据了 则不必继续递归
*/
int Extractor::forward_layer(int layer_index, bool inplace = false){
    Layer* p_layer = net->layers[layer_index];

    std::vector<Mat> bottom_mats{};

    for(int blob_index : p_layer->bottoms){
        // dims等于-1说明节点数据为空
        if(blob_mats[blob_index].empty()){
            int layer_index = net->blobs[blob_index].input;
            forward_layer(layer_index);
        }
        bottom_mats.push_back(blob_mats[blob_index]);
    }

    // p_layer层 产出的节点
    std::vector<Mat> top_mats(p_layer->tops.size());

    // 需要的节点都计算完毕了 forward
    if(p_layer->is_inplace){
        int ret = p_layer->forward_inplace(bottom_mats);
        // 把计算出来的结果写回到blob_mats里面  因为vector<>是值拷贝 所以必须手动回写
        for (size_t i = 0; i < p_layer->tops.size(); i++)
        {
            int top_blob_index = p_layer->tops[i]; // top 和 bottom是一样的值 但 是两个Blob

            blob_mats[top_blob_index] = bottom_mats[i];
        }
        return ret;
    }
    else{
        int ret = p_layer->forward(bottom_mats, top_mats);

        // 把计算出来的结果写回到blob_mats里面  因为vector<>是值拷贝 所以必须手动回写
        for (size_t i = 0; i < p_layer->tops.size(); i++)
        {
            int top_blob_index = p_layer->tops[i];

            blob_mats[top_blob_index] = top_mats[i];
        }


        return ret;
    }
}

int Extractor::extract(int index, Mat& v){
    // 计算第index节点 需要 哪个layer
    int layer_index = net->blobs[index].input;

    forward_layer(layer_index);

    v = blob_mats[index];

    return 0;
}

int Extractor::extract(string name, Mat& v){
    int index = net->find_blob_index_by_name(name);
    if(index == -1){
        NCNN_LOGE("There is no blob named %s, extract error!", name.c_str());
    }
    
    return extract(index, v);
}


}