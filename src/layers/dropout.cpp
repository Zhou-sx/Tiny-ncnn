#include "dropout.h"

namespace tiny_ncnn{

Layer* Dropout_layer_creator(void*){
    return new Dropout();
}

}