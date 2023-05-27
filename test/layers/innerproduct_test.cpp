#include <iostream>
#include <vector>

#include "innerproduct.h"
#include "packing.h"
#include "mat.h"
#include "paramdict.h"
#include "../tools.h"

using namespace tiny_ncnn;


int main(){
    /*
        a: 16
        b: 16 x 32
        c: 32
    */
    int cin = 128;
    int cout = 256;
    double gflops = cout * cin * 2 * 1.0e-09;

    Mat a = RandomMat({cin, 1, 1, 1}, 1);
    Mat b = RandomMat({cin, cout, 1, 1}, 2);
    std::cout << "a:" << std::endl;
    pretty_print(a);
    std::cout << "b:" << std::endl;
    pretty_print(b);

    // pack
    Layer* p_pack = Packing_layer_creator();
    ParamDict param;
    param.set(0, 4);
    p_pack->load_param(param);
    
    Mat b_pack;
    std::vector<Mat> v1 = {b};
    std::vector<Mat> v2 = {b_pack};
    p_pack->forward(v1, v2);
    b_pack = v2[0];
    
    std::cout << "b_pack:" << std::endl;
    pretty_print(b_pack);

    Layer* p_inner = InnerProduct_layer_creator();

    ParamDict param_inner;
    param_inner.set(0, cout);        // c_out
    param_inner.set(1, 0);           // bias_term
    param_inner.set(2, cin * cout);  // weight_data_size
    p_inner->load_param(param_inner);

    dynamic_cast<InnerProduct*>(p_inner)->weight_data_pack = b_pack;
    dynamic_cast<InnerProduct*>(p_inner)->weight_data = b;

    Mat c;
    std::vector<Mat> input = {a};
    std::vector<Mat> output = {c};
    p_inner->forward(input, output);

    c = output[0]; // 回写 否则m2不更新
    std::cout << "c: " << std::endl;
    pretty_print(c);

    time_test(p_inner, a, gflops);

    return 0;
}