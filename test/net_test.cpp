#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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
  // 图像输入
  cv::Mat img_bgr = cv::imread("/home/linaro/workspace/Tiny-ncnn/img/plane.jpg"), img_rgb;
  cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
  cv::Mat img_resize;
  cv::resize(img_rgb, img_resize, {227,227});
  Mat img = from_rgb_pixels(img_resize.data, 227, 227);
  float mean[3] = { 128.f, 128.f, 128.f };
  float norm[3] = { 1/128.f, 1/128.f, 1/128.f };
  img.substract_mean_normalize(mean, norm);

  Net net{};

  net.load_param("/home/linaro/workspace/Tiny-ncnn/model/alexnet.param");
  net.load_model("/home/linaro/workspace/Tiny-ncnn/model/alexnet.bin");
  Extractor extractor = net.create_extractor();
  extractor.input("data", img);

  Mat out;
  extractor.extract("fc6", out);

  std::cout << "output value of Net: " << std::endl;
  pretty_print(out.channel(0));
  
}