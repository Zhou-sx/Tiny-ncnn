#include "concat.h"


namespace tiny_ncnn{

int Concat::load_param(){
    return 0;
}

int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const{
    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;
    int d = bottom_blobs[0].d;
    int dims = bottom_blobs[0].dims;
    int elementsize = bottom_blobs[0].elemsize;
    int elementpack = bottom_blobs[0].elempack;

    // total channels
    int top_channels = 0;
    for (size_t b=0; b<bottom_blobs.size(); b++)
    {
        const Mat& bottom_blob = bottom_blobs[b];
        top_channels += bottom_blob.c;
    }

    Shape shape = {w, h, d, top_channels};
    Mat& top_blob = top_blobs[0];
    top_blob.create_buffer(shape, dims, elementsize, elementpack);
    if (top_blob.empty())
        return -1;

    int q = 0;
    for (size_t b=0; b<bottom_blobs.size(); b++)
    {
        const Mat& bottom_blob = bottom_blobs[b];

        int channels = bottom_blob.c;
        int size = bottom_blob.c_step * channels;

        const float* ptr = bottom_blob;
        float* outptr = top_blob.channel(q);
        for (int i=0; i<size; i++)
        {
            outptr[i] = ptr[i];
        }

        q += channels;
    }
    return 0;
}

Layer* Concat_layer_creator(void* /*userdata*/){
    return new(Concat);
}

}