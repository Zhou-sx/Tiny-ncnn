#include "net.h"
#include "extractor.h"
#include "paramdict.h"
#include "layer_factory.h"
#include "platform.h"

namespace tiny_ncnn{

Extractor Net::create_extractor(){
    return Extractor(this);
}

int Net::find_blob_index_by_name(std::string _name) const{
    for(int i = 0; i < blobs.size(); i++){
        if(blobs[i].name == _name){
            return i;
        }
    }
    return -1;
}

int Net::find_layer_index_by_name(std::string _name) const{
    for(int i = 0; i < layers.size(); i++){
        if(layers[i]->name == _name){
            return i;
        }
    }
    return -1;
}

int Net::load_param(const DataReader& dr){
#define SCAN_VALUE(fmt, v)                \
    if (dr.scan(fmt, &v) != 1)            \
    {                                     \
        NCNN_LOGE("parse " #v " failed"); \
        return -1;                        \
    }

    int magic = 0;
    SCAN_VALUE("%d", magic);
    if (magic != 7767517)
    {
        NCNN_LOGE("param is too old, please regenerate");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        NCNN_LOGE("invalid layer_count or blob_count");
        return -1;
    }

    layers.resize((size_t)layer_count);
    blobs.resize((size_t)blob_count);

    ParamDict pd;

    int blob_index = 0; // blob命名编号从1开始
    // 逐层
    for (int i = 0; i < layer_count; i++)
    {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)

        Layer* layer = create_layer(layer_type);
        if (!layer)
        {
            NCNN_LOGE("layer %s not exists or registered", layer_type);
            return -1;
        }
        layer->name = layer_name;
        
        // 层中 bottom_blob
        layer->bottoms.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++){
            char bottom_name[256];
            SCAN_VALUE("%255s", bottom_name)
            
            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            // 如果之前没找到
            if (bottom_blob_index == -1){
                Blob& blob = blobs[blob_index];
                bottom_blob_index = blob_index;
                blob.name = bottom_name;
                blob_index++;
            }

            Blob& blob = blobs[bottom_blob_index];
            

            layer->bottoms[j] = bottom_blob_index;
        }

        // 层中 top_blob
        layer->tops.resize(top_count);
        for (int j = 0; j < top_count; j++){
            char top_name[256];
            SCAN_VALUE("%255s", top_name);

            // 输出的节点一定是新的blob 否则就存在环
            Blob& blob = blobs[blob_index];
            blob.name = top_name;
            blob.input = i; // blob由第i层产生

            layer->tops[j] = blob_index;
            blob_index++;
        }

        // 层中剩下的参数 ? = ?
        int ret = pd.load_param(dr);
        if (ret != 0)
        {
            NCNN_LOGE("ParamDict load_param %d %s failed", i, layer->name.c_str());
            continue;
        }

        // 读取完了所有参数 初始化层
        layer->load_param(pd);

        // 更新layers列表
        layers[i] = layer;
    }

    return 0;
}

int Net::load_param(const char* path){
    FILE* fp = fopen(path, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", path);
        return -1;
    }
    DataReader dr(fp);
    int ret = load_param(dr);
    fclose(fp);
    return ret;
}

int Net::load_model(const DataReader& dr){
    ModelBin mb(dr);
    for(int i=0; i<layers.size(); i++){
        layers[i]->load_model(mb);
    }

    return 0;
}

int Net::load_model(const char* path){
    FILE* fp = fopen(path, "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen %s failed", path);
        return -1;
    }
    DataReader dr(fp);
    int ret = load_model(dr);
    fclose(fp);
    return ret;
}

}