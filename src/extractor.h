#ifndef NCNN_EXTRACTOR_H
#define NCNN_EXTRACTOR_H

#include "mat.h"
#include "net.h"

namespace tiny_ncnn{

class Net;

class Extractor{
public:
    Extractor() = default;
    Extractor(Net* const _net);

public:
    int input(string name, const Mat& v);
    int input(int index, const Mat& v);

    int extract(string name, Mat& v);
    int extract(int index, Mat& v);

    int forward_layer(int layer_index, bool inplace); // 检查 layer_index 层是否准备就绪 推理

private:
    vector<Mat> blob_mats;
    tiny_ncnn::Net* const net;

};

}


#endif