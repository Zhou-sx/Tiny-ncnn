#ifndef NCNN_InnerProduct_LAYER_H
#define NCNN_InnerProduct_LAYER_H
#include "../layer.h"


namespace tiny_ncnn{

class InnerProduct : public Layer{
public:
    InnerProduct() = default;

    int load_param(const ParamDict&);
    int load_model(const ModelBin&);

    void set_model(){
        weight_data = Mat(Shape(27,1,1,1), 1, 4);
        
        for(int i=0; i<27; i++){
            weight_data[i] = 1;
        }

        bias_data = Mat(Shape(3,1,1,1), 1, 4);

    }

    int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

public:
    int c_out;
    int bias_term;
    int weight_data_size;

    // model
    Mat weight_data;
    Mat bias_data;
};

Layer* InnerProduct_layer_creator(void* = nullptr);

}

#endif