#ifndef NCNN_Conv3x3s1_H
#define NCNN_Conv3x3s1_H
#include <arm_neon.h>

#include "../mat.h"

namespace tiny_ncnn{

/*
    kenel_size = (3,3)
    stride = (1,1)
    dilation = (1,1)
*/
int conv3x3s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias){
    const int inw = bottom_blob.w;
    const int inh = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    // kernel/bias是flatten的
    const float* kernel = (float*)_kernel;
    const float* bias = (float*)_bias;


    for(int p = 0; p < outch; p++){
        // 设置bias
        const float bias0 = bias ? bias[p] : 0.f;
        Mat out = top_blob.channel(p);
        out.fill(bias0);

        for(int q = 0; q < inch; q++){
            const float* data_in = (float*)bottom_blob.channel(q);
            float* data_out = (float*)out;

            for(int h = 0; h < outh; h++){
                // 一次求特征图上4点
                int w_patch = outw >> 2;
                int w_remain = outw << 2;

                // 准备卷积核
                float32x4_t k0_0 = vld1q_f32(kernel);
                float32x4_t k0_1 = vld1q_f32(kernel + 3);
                float32x4_t k0_2 = vld1q_f32(kernel + 6);

                // 准备数据
                const float* r0 = data_in;
                const float* r1 = data_in + inw;
                const float* r2 = data_in + inw * 2;

                // 输出数据偏移
                float* t = data_out + outw * h;
                for(int w = 0; w < w_patch; w += 4){
                    // 读数据
                    float32x4_t sum = vld1q_f32(t);

                    float32x4x2_t _r00 = vld2q_f32(r0);
                    float32x4x2_t _r03 = vld2q_f32(r0 + 8);
                    float32x4_t _r013 = vextq_f32(_r00.val[0], _r03.val[0], 1);
                    float32x4_t _r016 = vextq_f32(_r00.val[0], _r03.val[0], 2);

                    float32x4x2_t _r10 = vld2q_f32(r1);
                    float32x4x2_t _r13 = vld2q_f32(r1 + 8);
                    float32x4_t _r113 = vextq_f32(_r10.val[0], _r13.val[0], 1);
                    float32x4_t _r116 = vextq_f32(_r10.val[0], _r13.val[0], 2);

                    float32x4x2_t _r20 = vld2q_f32(r2);
                    float32x4x2_t _r23 = vld2q_f32(r2 + 8);
                    float32x4_t _r213 = vextq_f32(_r20.val[0], _r23.val[0], 1);
                    float32x4_t _r216 = vextq_f32(_r20.val[0], _r23.val[0], 2);

                    sum = vmlaq_laneq_f32(sum, _r00.val[0], k0_0, 0);
                    sum = vmlaq_laneq_f32(sum, _r013, k0_0, 1);
                    sum = vmlaq_laneq_f32(sum, _r013, k0_0, 2);

                    sum = vmlaq_laneq_f32(sum, _r10.val[0], k0_1, 0);
                    sum = vmlaq_laneq_f32(sum, _r113, k0_1, 1);
                    sum = vmlaq_laneq_f32(sum, _r116, k0_1, 2);

                    sum = vmlaq_laneq_f32(sum, _r20.val[0], k0_2, 0);
                    sum = vmlaq_laneq_f32(sum, _r213, k0_2, 1);
                    sum = vmlaq_laneq_f32(sum, _r216, k0_2, 2);

                    // 写数据
                    vst1q_f32(t, sum);
                    t += 4;

                    // 移动输入数据
                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                }
                // 如果不是4的倍数
                for(int w = w_remain; w < outw; w++){
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r03 = vld1q_f32(r1);
                    float32x4_t _r06 = vld1q_f32(r2);

                    float32x4_t _sum0 = vmulq_f32(_r00, k0_0);
                    _sum0 = vmlaq_f32(_sum0, _r03, k0_1);
                    _sum0 = vmlaq_f32(_sum0, _r06, k0_2);
                    
                    // 把_sum0的第四个通道设为历史值
                    _sum0 = vsetq_lane_f32(*t, _sum0, 3);
                    // 累加
                    *t = vaddvq_f32(_sum0);
                    t++;

                    r0++;
                    r1++;
                    r2++;
                }
            
            }
        
            kernel += 9;
        }
    }
    return 0;
}

}


#endif