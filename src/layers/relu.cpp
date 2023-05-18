#include "relu.h"

namespace tiny_ncnn{

int ReLU::forward_inplace(std::vector<Mat>& bottom_blobs) const{
    for(int i=0; i<bottom_blobs.size(); i++){
        Mat bottom_top_blob = bottom_blobs[i];
        size_t size = bottom_top_blob.c_step;

        for (int q = 0; q < bottom_top_blob.c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }

    }
    
    return 0;
}

Layer* ReLU_layer_creator(void*){
    return new ReLU();
}

}