#include "conv.h"
#include <assert.h>

namespace tiny_ncnn{
/*
    从.param文件中读取层参数
*/
int Convolution::load_param(const ParamDict& pd){
    c_out = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);

    return 0;
}

/*
    从.bin文件中读取权值
*/

int Convolution::load_model(const ModelBin& mb){
    weight_data = mb.load(weight_data_size, 0);
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

/*  
    bottom_blob: [w_in,  h_in,  c_in]
    top_blob:    [w_out, h_out, c_out]
*/
int Convolution::do_forward(const Mat& bottom_blob, Mat& top_blob) const{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }
    
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                const float* kptr = (const float*)weight_data + maxk * inch * p; // 卷积核是flatten的
                // const float* kptr = (const float*)weight_data.channel(p);     // 卷积核是Mat

                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row<float>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++) 
                    {
                        float val = sptr[space_ofs[k]]; 
                        float wt = kptr[k];
                        sum += val * wt; 
                    }

                    kptr += maxk;
                }

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }

    return 0;
}

/*
    Convolution 2D:
        bottom_blobs 和 top_blobs 有且只有一个 不支持多输入多输出
*/
int Convolution::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const{
    assert(bottom_blobs.size() == 1);
    assert(top_blobs.size() == 1);
    assert(bottom_blobs[0].dims == 3);

    Mat bottom_border_blob;
    copy_make_border(bottom_blobs[0], bottom_border_blob, pad_top, pad_bottom, pad_left, pad_right, pad_value);

    const int w = bottom_border_blob.w;
    const int h = bottom_border_blob.h;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;
    top_blobs[0] = Mat(Shape(outw, outh, 1, c_out), 3, 4);
    do_forward(bottom_border_blob, top_blobs[0]);

    return 0;
}

Layer* Convolution_layer_creator(void*){
    return new Convolution();
}

}