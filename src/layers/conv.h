#ifndef NCNN_Convolution_LAYER_H
#define NCNN_Convolution_LAYER_H
#include "../layer.h"

namespace tiny_ncnn{

class Convolution :public Layer{
public:
    Convolution() = default;

    int load_param(const ParamDict&);
    int load_model(const ModelBin&);

    int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;
    int do_forward(const Mat& bottom_blobs, Mat& top_blobs) const;


    void set_model(){
        weight_data = Mat(Shape(3,3,3,5), 4, 4);
        int cnt = 0;
        for(int c=0; c<5; c++){
            Mat m = weight_data.channel(c);
            for(int i=0; i<m.total(); i++){
                m[i] = cnt++; 
            }
        }

        bias_data = Mat(Shape(5,1,1,1), 1, 4);

    }

private:
    int c_out;        // 0
    
    int kernel_w;      // 1
    int kernel_h;      // 11
    
    int dilation_w;   // 2
    int dilation_h;   // 12
    
    int stride_w;     // 3
    int stride_h;     // 13

    int pad_left;     // 4
    int pad_top;      // 14
    int pad_right;    // 15
    int pad_bottom;   // 16
    int pad_value;    // 18

    int bias_term;    // 5
    int weight_data_size; // 6


    // model 
    Mat weight_data;
    Mat bias_data;
};

Layer* Convolution_layer_creator(void* = nullptr);

}
#endif