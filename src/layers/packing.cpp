#include <assert.h>
#include <arm_neon.h>

#include "packing.h"

#define __ARM_NEON 1

namespace tiny_ncnn{

int Packing::load_param(const ParamDict& pd){
    out_elempack = pd.get(0, 1);
    return 0;
};

int Packing::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const{
    assert(bottom_blobs.size() == 1);
    assert(top_blobs.size() == 1);

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // 不用转换
    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;

    // 暂时只提供 pack 4
    assert(pack1to4 || pack4to1);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        /*
            w => w/out_elempack
        */
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.c_step = w * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        /*
            w x h => w x h/out_elempack
        */
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create_buffer({w, outh, 1, 1}, 2, out_elemsize);
        top_blob.elempack = out_elempack;
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row<float>(i * 4);
                const float* r1 = bottom_blob.row<float>(i * 4 + 1);
                const float* r2 = bottom_blob.row<float>(i * 4 + 2);
                const float* r3 = bottom_blob.row<float>(i * 4 + 3);

                float* outptr = top_blob.row<float>(i);

                int j = 0;
            #if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {   
                    /*
                        vst4q_f32 交叉存储
                        _p[0][0] => outptr[0]
                        _p[1][0] => outptr[1]
                        _p[2][0] => outptr[2]
                        _p[3][0] => outptr[3]
                        
                        ... ...
                    */
                    float32x4x4_t _p;
                    _p.val[0] = vld1q_f32(r0);
                    _p.val[1] = vld1q_f32(r1);
                    _p.val[2] = vld1q_f32(r2);
                    _p.val[3] = vld1q_f32(r3);
                    vst4q_f32(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
            #endif
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row<float>(i);

                float* outptr0 = top_blob.row<float>(i * 4);
                float* outptr1 = top_blob.row<float>(i * 4 + 1);
                float* outptr2 = top_blob.row<float>(i * 4 + 2);
                float* outptr3 = top_blob.row<float>(i * 4 + 3);

                int j = 0;
            #if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    /*
                        vst4q_f32 交叉加载
                        r0[0] => _p[0][0]
                        r0[1] => _p[1][0]
                        r0[2] => _p[2][0]
                        r0[3] => _p[3][0]
                        
                        ... ...
                    */
                    float32x4x4_t _p = vld4q_f32(r0);
                    vst1q_f32(outptr0, _p.val[0]);
                    vst1q_f32(outptr1, _p.val[1]);
                    vst1q_f32(outptr2, _p.val[2]);
                    vst1q_f32(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
            #endif
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        /*
            dims 3:
            w x h x c => w x h x c/out_elempack

            dims 4:
            w x h x d x c => w x h x d x c/out_elempack

        */
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create_buffer({w, h, 1, outc}, 3, out_elemsize);
        else // if (dims == 4)
            top_blob.create_buffer({w, h, d, outc}, 4, out_elemsize);
        top_blob.elempack = out_elempack;

        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 4);
                const float* r1 = bottom_blob.channel(q * 4 + 1);
                const float* r2 = bottom_blob.channel(q * 4 + 2);
                const float* r3 = bottom_blob.channel(q * 4 + 3);

                float* outptr = top_blob.channel(q);

                int i = 0;
            #if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _p;
                    _p.val[0] = vld1q_f32(r0);
                    _p.val[1] = vld1q_f32(r1);
                    _p.val[2] = vld1q_f32(r2);
                    _p.val[3] = vld1q_f32(r3);
                    vst4q_f32(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
            #endif
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

                int i = 0;
            #if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _p = vld4q_f32(r0);
                    vst1q_f32(outptr0, _p.val[0]);
                    vst1q_f32(outptr1, _p.val[1]);
                    vst1q_f32(outptr2, _p.val[2]);
                    vst1q_f32(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
            #endif
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    return 0;
};

Layer* Packing_layer_creator(void*){
    return new Packing();
}

}