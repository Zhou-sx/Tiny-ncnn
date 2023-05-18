#include <assert.h>
#include <cmath>

#include "lrn.h"

namespace tiny_ncnn{

int LRN::load_param(const ParamDict& pd)
{
    region_type = pd.get(0, 0);
    local_size = pd.get(1, 5);
    alpha = pd.get(2, 1.f);
    beta = pd.get(3, 0.75f);
    bias = pd.get(4, 1.f);

    return 0;
}

/*
    输入一个节点
*/
int LRN::forward_inplace(std::vector<Mat>& bottom_blobs) const{
    assert(bottom_blobs.size() == 1);

    Mat bottom_blob = bottom_blobs[0];

    // 预处理 平方
    int channels = bottom_blob.c;
    size_t size = bottom_blob.c_step;

    Mat square_blob(bottom_blob.shape, bottom_blob.dims, bottom_blob.elemsize);
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = square_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            outptr[i] = ptr[i] * ptr[i];
        }
    }

    if(region_type == 0){
        Mat square_sum(bottom_blob.shape, bottom_blob.dims, bottom_blob.elemsize);
        square_sum.fill(0.f);

        const float alpha_div_size = alpha / local_size;
        for (int q = 0; q < channels; q++)
        {
            // square sum
            float* ssptr = square_sum.channel(q);
            for (int p = q - local_size / 2; p <= q + local_size / 2; p++)
            {
                if (p < 0 || p >= channels)
                    continue;

                const float* sptr = square_blob.channel(p);
                for (int i = 0; i < size; i++)
                {
                    ssptr[i] += sptr[i];
                }
            }

            float* ptr = bottom_blob.channel(q);
            for (int i = 0; i < size; i++)
            {
                ptr[i] = static_cast<float>(ptr[i] * std::pow(bias + alpha_div_size * ssptr[i], -beta));
            }
        }

    }
    else if(region_type == 1){
        /*
            skip
        */
    }

    return 0;
}

Layer* LRN_layer_creator(void*){
    return new LRN();
}

}