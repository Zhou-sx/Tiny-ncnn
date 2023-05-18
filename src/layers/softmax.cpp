#include <assert.h>
#include <cmath>
#include <float.h>

#include "softmax.h"

namespace tiny_ncnn{

int Softmax::load_param(const ParamDict& pd){
    axis = pd.get(0, 0);
    return 0;
}

/*
   value = exp( value - global max value )
   sum all value
   value = value / sum
*/
int Softmax::forward_inplace(std::vector<Mat>& bottom_blobs) const{
    assert(bottom_blobs.size() == 1);

    Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        int w = bottom_blob.w;

        float* ptr = bottom_blob;

        float max = -FLT_MAX;
        for (int i = 0; i < w; i++)
        {
            max = std::max(max, ptr[i]);
        }

        float sum = 0.f;
        for (int i = 0; i < w; i++)
        {
            ptr[i] = static_cast<float>(exp(ptr[i] - max));
            sum += ptr[i];
        }

        for (int i = 0; i < w; i++)
        {
            ptr[i] /= sum;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        Mat max({w, 1, 1, 1}, 1, elemsize);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);

        // 遍历每一行 找到列中最大的数值
        for (int i = 0; i < h; i++)
        {
            const float* ptr = bottom_blob.row<float>(i);
            for (int j = 0; j < w; j++)
            {
                max[j] = std::max(max[j], ptr[j]);
            }
        }

        Mat sum({w, 1, 1, 1}, 1, elemsize);
        if (sum.empty())
            return -100;
        sum.fill(0.f);

        // 遍历每一行 得到每一列的累加值
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_blob.row<float>(i);
            for (int j = 0; j < w; j++)
            {
                ptr[j] = static_cast<float>(exp(ptr[j] - max[j]));
                sum[j] += ptr[j];
            }
        }

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_blob.row<float>(i);
            for (int j = 0; j < w; j++)
            {
                ptr[j] /= sum[j];
            }
        }
    }
    
    if (dims == 2 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_blob.row<float>(i);
            float m = -FLT_MAX;
            for (int j = 0; j < w; j++)
            {
                m = std::max(m, ptr[j]);
            }

            float s = 0.f;
            for (int j = 0; j < w; j++)
            {
                ptr[j] = static_cast<float>(exp(ptr[j] - m));
                s += ptr[j];
            }

            for (int j = 0; j < w; j++)
            {
                ptr[j] /= s;
            }
        }
    }

    if (dims == 3 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        Mat max({w, h, 1, 1}, 2, elemsize);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);
        // 取每个通道的最大值
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                max[i] = std::max(max[i], ptr[i]);
            }
        }

        Mat sum({w, h, 1, 1}, 2, elemsize);
        if (sum.empty())
            return -100;
        sum.fill(0.f);
        // 每个通道的累加值
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                ptr[i] = static_cast<float>(exp(ptr[i] - max[i]));
                sum[i] += ptr[i];
            }
        }

        // softmax值
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                ptr[i] /= sum[i];
            }
        }
    }

    if (dims == 3 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        Mat max({w, channels, 1, 1}, 2, elemsize);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* maxptr = max.row<float>(q);

            // 每一列中最大的
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    maxptr[j] = std::max(maxptr[j], ptr[j]);
                }

                ptr += w;
            }
        }

        Mat sum({w, channels, 1, 1}, 2, elemsize);
        if (sum.empty())
            return -100;
        sum.fill(0.f);
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_blob.channel(q);
            float* maxptr = max.row<float>(q);
            float* sumptr = sum.row<float>(q);

            // 每一列中累加值
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = static_cast<float>(exp(ptr[j] - maxptr[j]));
                    sumptr[j] += ptr[j];
                }

                ptr += w;
            }
        }

        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_blob.channel(q);
            float* sumptr = sum.row<float>(q);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    ptr[j] /= sumptr[j];
                }

                ptr += w;
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                float max = -FLT_MAX;
                for (int j = 0; j < w; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                float sum = 0.f;
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = static_cast<float>(exp(ptr[j] - max));
                    sum += ptr[j];
                }

                for (int j = 0; j < w; j++)
                {
                    ptr[j] /= sum;
                }

                ptr += w;
            }
        }
    }
    return 0;
}


Layer* Softmax_layer_creator(void*){
    return new Softmax();
}

}