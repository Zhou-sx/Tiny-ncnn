#ifndef NCNN_LAYER_FACTORY_H
#define NCNN_LAYER_FACTORY_H

#include <functional>
#include <unordered_map>

#include "layer.h"
#include "layers/absval.h"
#include "layers/split.h"
#include "layers/concat.h"
#include "layers/padding.h"
#include "layers/conv.h"
#include "layers/input.h"
#include "layers/relu.h"
#include "layers/pooling.h"
#include "layers/lrn.h"
#include "layers/innerproduct.h"
#include "layers/dropout.h"
#include "layers/convolutiondepthwise.h"
#include "layers/softmax.h"

namespace tiny_ncnn{

extern std::unordered_map<std::string, int> layer_name_to_index;
extern std::function<Layer*(void*)> layer_registry[];

int layer_to_index(const std::string& type);

Layer*  create_layer(int idx);

Layer* create_layer(const std::string& type);


}

#endif