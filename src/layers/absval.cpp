#include "absval.h"

namespace tiny_ncnn{


int Absval::forward_inplace(std::vector<Mat>& bottom_blobs) const{
    for(int i=0; i<bottom_blobs.size(); i++){
        Mat bottom_top_blob = bottom_blobs[i];
        size_t size = bottom_top_blob.c_step;

        for (int q = 0; q < bottom_top_blob.c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = -ptr[i];
            }
        }

    }
    return 0;
}

Layer* Absval_layer_creator(void* = nullptr /*userdata*/){
    return new(Absval);
}

}