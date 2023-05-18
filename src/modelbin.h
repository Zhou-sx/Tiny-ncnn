#ifndef NCNN_MODELBIN_H
#define NCNN_MODELBIN_H

#include "mat.h"
#include "data_reader.h"

namespace tiny_ncnn{

class ModelBin{
public:
    ModelBin() = default;
    ModelBin(const DataReader& _dr): dr(_dr) {};

    // element type
    // 0 = auto
    // 1 = float32
    Mat load(int w, int type) const;

    const DataReader& dr; // ModelBin中的DataReader是全局的

};

}

#endif