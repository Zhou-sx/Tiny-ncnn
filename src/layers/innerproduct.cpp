#include <assert.h>

#include "innerproduct.h"

namespace tiny_ncnn{

int InnerProduct::load_param(const ParamDict& pd){
    c_out = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    return 0;
}

int InnerProduct::load_model(const ModelBin& mb){
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

    return 0;
}

Layer* InnerProduct_layer_creator(void*){
    return new InnerProduct();
}

}