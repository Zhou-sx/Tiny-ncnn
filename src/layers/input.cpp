#include "input.h"

namespace tiny_ncnn{

int Input::load_param(const ParamDict& pd){
    w = pd.get(0, 1);
    h = pd.get(1, 1);
    c = pd.get(2, 1);
    d = pd.get(3, 1);

    return 0;
}

Layer* Input_layer_creator(void*){
    return new Input();
}

}