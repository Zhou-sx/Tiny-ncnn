#include <arm_neon.h>
#include <assert.h>
#include <vector>

#include "innerproduct.h"
#include "../layer_factory.h"
#include "../../test/tools.h"

#define __ARM_NEON 1

namespace tiny_ncnn{

int InnerProduct::load_param(const ParamDict& pd){
    c_out = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    return 0;
}

int InnerProduct::load_model(const ModelBin& mb){
    weight_data = mb.load(weight_data_size, 0);
    #if __ARM_NEON
            /*
                1. weight_data的Reshape:   c_in*c_out   => c_in x c_out (c_in = h*w*c)
                2. pack:                   c_in x c_out => c_in x c_out/4
            */
        const int c_in = weight_data_size / c_out;
        Mat weight_data_r2 = weight_data.reshape({c_in, c_out, 1, 1}, 2);

        Layer* p = Packing_layer_creator();
        ParamDict param;
        param.set(0, 4);
        p->load_param(param);
        
        std::vector<Mat> v1 = {weight_data_r2};
        std::vector<Mat> v2 = {weight_data_pack};
        p->forward(v1, v2);
        weight_data_pack = v2[0];


    #endif

    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(c_out, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int InnerProduct::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const{
    assert(bottom_blobs.size() == 1);
    assert(top_blobs.size() == 1);

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    const int c_in = weight_data_size / c_out;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    if (bottom_blob.dims == 2 && w == c_in && h > 1)
    {
        // gemm
        top_blob = Mat({c_out, h, 1, 1}, 2, elemsize);
        if (top_blob.empty())
            return -100;

        for (int j = 0; j < h; j++)
        {
            const float* m = bottom_blob.row<float>(j);
            float* outptr = top_blob.row<float>(j);

            for (int p = 0; p < c_out; p++)
            {
                const float* kptr = (const float*)weight_data + w * p;

                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                for (int i = 0; i < w; i++)
                {
                    sum += m[i] * kptr[i];
                }

                outptr[p] = sum;
            }
        }

        return 0;
    }
    
    /*
        bottom_blob.dims == 1的特殊情况
    */
    #ifdef __ARM_NEON

    // flatten 展平
    const Mat bottom_blob_flatten = bottom_blob.reshape({c_in, 1, 1, 1}, 1);

    top_blob= Mat({c_out / 4, 1, 1, 1}, 1, elemsize*4, 4);
    if (top_blob.empty())
        return -100;

    float* bias_ptr = static_cast<float*>(bias_data.data);
    float* out_ptr = static_cast<float*>(top_blob.data);

    for (int p = 0; p < c_out / 4; p++){
        float* a_ptr = static_cast<float*>(bottom_blob_flatten.data);
        float* b_ptr = static_cast<float*>(weight_data_pack.channel(p).data);
        // std::cout << "weight_data_pack[p]:" << std::endl;
        // pretty_print(weight_data_pack.channel(p));

        float32x4_t sum_0;
        if(bias_term){
            sum_0 = vld1q_f32(bias_ptr);
            bias_ptr += 4;
        }
        else{
            sum_0 = vdupq_n_f32(0);
        }

        float32x4_t sum_1 = vdupq_n_f32(0);
        float32x4_t sum_2 = vdupq_n_f32(0);
        float32x4_t sum_3 = vdupq_n_f32(0);

        int k = 0;
        for(; k < c_in; k += 8){
            asm volatile(
                "prfm       pldl1keep, [%0, #256]     \n"
                "ld1        {v0.4s, v1.4s}, [%0], #32 \n"
                "prfm       pldl1keep, [%1, #512]     \n"
                "ld1        {v2.4s, v3.4s, v4.4s, v5.4s}, [%1], #64 \n"
                "prfm       pldl1keep, [%1, #512]     \n"
                "ld1        {v6.4s, v7.4s, v8.4s, v9.4s}, [%1], #64 \n"

                // 虽然存在目的寄存器冲突 WAW相关 但是不影响性能
                "fmla       %2.4s, v2.4s, v0.s[0]     \n"
                "fmla       %3.4s, v3.4s, v0.s[1]     \n"
                "fmla       %4.4s, v4.4s, v0.s[2]     \n"
                "fmla       %5.4s, v5.4s, v0.s[3]     \n"

                "fmla       %2.4s, v6.4s, v1.s[0]     \n"
                "fmla       %3.4s, v7.4s, v1.s[1]     \n"
                "fmla       %4.4s, v8.4s, v1.s[2]     \n"
                "fmla       %5.4s, v9.4s, v1.s[3]     \n"
                : "=r"(a_ptr),  // %0
                "=r"(b_ptr),  // %1
                "=w"(sum_0), // %2
                "=w"(sum_1), // %3
                "=w"(sum_2), // %4
                "=w"(sum_3)  // %5
                : "0"(a_ptr),
                "1"(b_ptr),
                "2"(sum_0),
                "3"(sum_1),
                "4"(sum_2),
                "5"(sum_3)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
        }

        // 不能被8整除
        for(; k < c_in; k += 4){
            float32x4_t a = vld1q_f32(a_ptr);
            a_ptr += 4;

            float32x4_t b_0 = vld1q_f32(b_ptr);
            b_ptr += 4;
            float32x4_t b_1 = vld1q_f32(b_ptr);
            b_ptr += 4;
            float32x4_t b_2 = vld1q_f32(b_ptr);
            b_ptr += 4;
            float32x4_t b_3 = vld1q_f32(b_ptr);
            b_ptr += 4;

            sum_0 = vfmaq_laneq_f32(sum_0, b_0, a, 0);
            sum_1 = vfmaq_laneq_f32(sum_1, b_1, a, 1);
            sum_2 = vfmaq_laneq_f32(sum_2, b_2, a, 2);
            sum_3 = vfmaq_laneq_f32(sum_3, b_3, a, 3);
        }

        // 不能被4整除
        for(; k < c_in; k ++){
            float32_t a = *a_ptr;
            float32x4_t b_0 = vld1q_f32(b_ptr);
            sum_0 = vmlaq_n_f32(sum_0, b_0, a);

            a_ptr += 1;
            b_ptr += 4;
        }

        sum_0 = vaddq_f32(sum_0, sum_1);
        sum_2 = vaddq_f32(sum_2, sum_3);
        sum_0 = vaddq_f32(sum_0, sum_2);

        vst1q_f32(out_ptr, sum_0);
        out_ptr += 4;
    }

    // unpack
    
    Mat top_blob_unpack = Mat({c_out, 1, 1, 1}, 1, elemsize, 1);

    Layer* p = Packing_layer_creator();
    ParamDict param;
    param.set(0, 1);
    p->load_param(param);
    
    std::vector<Mat> v1 = {top_blob};
    std::vector<Mat> v2 = {top_blob_unpack};
    p->forward(v1, v2);
    top_blob = v2[0];

    #else
    top_blob= Mat({c_out, 1, 1, 1}, 1, elemsize);
    if (top_blob.empty())
        return -100;

    for (int p = 0; p < c_out; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* w = (const float*)weight_data + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        top_blob[p] = sum;
    }
    #endif

    return 0;
}

Layer* InnerProduct_layer_creator(void*){
    return new InnerProduct();
}

}