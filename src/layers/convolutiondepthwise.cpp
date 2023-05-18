#include <assert.h>

#include "convolutiondepthwise.h"

namespace tiny_ncnn{

int ConvolutionDepthWise::load_param(const ParamDict& pd){
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
    group = pd.get(7, 1);

    return 0;
}

int ConvolutionDepthWise::load_model(const ModelBin& mb){
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

int ConvolutionDepthWise::do_forward(const Mat& bottom_blob, Mat& top_blob) const{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

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

    // depth-wise
    if (inch == group && group == outch)
    {
        // 每一个通道 使用一个卷积核
        for (int g = 0; g < group; g++)
        {
            float* outptr = top_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            const Mat m = bottom_blob.channel(g);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[g];

                    const float* sptr = m.row<float>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        float w = kptr[k];
                        sum += val * w;
                    }

                    outptr[j] = sum;
                }

                outptr += outw;
            }
        }
    }
    else{
        // group convolution
        const int inch_g = inch / group;
        const int outch_g = outch / group;
        for (int g = 0; g < group; g++)
        {
            for (int p = 0; p < outch_g; p++)
            {
                float* outptr = top_blob.channel(g * outch_g + p);
                const float* weight_data_ptr = (const float*)weight_data + maxk * inch_g * outch_g * g;

                // shadowed variable for less openmp task args
                const int outw = top_blob.w;
                const int outh = top_blob.h;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                            sum = bias_data[outch_g * g + p];

                        const float* kptr = weight_data_ptr + maxk * inch_g * p;

                        for (int q = 0; q < inch_g; q++)
                        {
                            const Mat m = bottom_blob.channel(inch_g * g + q);
                            const float* sptr = m.row<float>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }
    
    return 0;
}

int ConvolutionDepthWise::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const{
    assert(bottom_blobs.size() == 1);
    assert(top_blobs.size() == 1);

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


Layer* ConvolutionDepthWise_layer_creator(void*){
    return new ConvolutionDepthWise();
}

}