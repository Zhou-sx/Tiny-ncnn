#include <iostream>

#include "net.h"
#include "layer_factory.h"
#include "tools.h"

using namespace tiny_ncnn;

/*
    节点：
    0     1
    |     |
    |     |
    2     3
     \   /
      \ /
       4
       |
       |
       5
      / \
     /   \
    6     7

    边
   0|     |1
    |     |
    2\   /
      \ /
      3|
       |  
     4/ \5
     /   \

*/
Layer* make_layer(int layer_type, vector<int> _bottoms, vector<int> _tops){
  Layer* p_layer = layer_registry[layer_type](nullptr);
  p_layer->bottoms = _bottoms;
  p_layer->tops = _tops;
  return p_layer;
}

int main(){
  Mat m({227, 227, 1, 3}, 3, 4);
  Mat m1 = m.channel(0);
  m1.fill(0.0f);
  Mat m2 = m.channel(1);
  m2.fill(1.0f);
  Mat m3 = m.channel(2);
  m3.fill(1.0f);

  Net net{};

  net.load_param("/mnt/c/Users/lenovo/Desktop/Code/Tiny-ncnn/model/alexnet.param");
  net.load_model("/mnt/c/Users/lenovo/Desktop/Code/Tiny-ncnn/model/alexnet.bin");
  Extractor extractor = net.create_extractor();
  extractor.input("data", m);

  Mat out;
  extractor.extract("prob", out);

  std::cout << "output value of Net: " << std::endl;
  pretty_print(out.channel(0));
  
}