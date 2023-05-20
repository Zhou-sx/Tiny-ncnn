#include <assert.h>
#include <float.h>

#include "pooling.h"

namespace tiny_ncnn{

int Pooling::load_param(const ParamDict& pd){
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    global_pooling = pd.get(4, 0);
    return 0;
}

int Pooling::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const{
    assert(bottom_blobs.size() == 1);
    assert(top_blobs.size() == 1);
    assert(bottom_blobs[0].dims == 3);

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];


    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;


    Mat bottom_blob_bordered;
    make_padding(bottom_blobs[0], bottom_blob_bordered);
    
    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;
    top_blob = Mat({outw, outh, 1, channels}, 3, elemsize);

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == 0)
    {
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row<float>(i * stride_h) + j * stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == 1)
    {
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                int sy0 = i * stride_h;

                for (int j = 0; j < outw; j++)
                {
                    int sx0 = j * stride_w;

                    float sum = 0;
                    int area = 0;

                    for (int ki = 0; ki < kernel_h; ki++)
                    {
                        int sy = sy0 + ki;

                        if (sy < pad_top)
                            continue;

                        if (sy >= h - pad_bottom)
                            break;

                        for (int kj = 0; kj < kernel_w; kj++)
                        {
                            int sx = sx0 + kj;

                            if (sx < pad_left)
                                continue;

                            if (sx >= w - pad_right)
                                break;

                            float val = m.row<float>(sy)[sx];
                            sum += val;
                            area += 1;
                        }
                    }

                    outptr[j] = sum / area;
                }

                outptr += outw;
            }
        }
    }
    return 0;
}

void Pooling::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered) const{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (pooling_type == 0)
    {
        pad_value = bottom_blob.elemsize == 1 ? -128.f : -FLT_MAX;
    }
    else if (pooling_type == 1)
    {
        pad_value = 0.f;
    }

    // 还可以继续细化 由三种 padding的种类 full same valid
    copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, pad_value);

}

Layer* Pooling_layer_creator(void*){
    return new Pooling();
}

}