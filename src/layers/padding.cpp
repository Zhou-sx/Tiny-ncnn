#include "padding.h"
#include "string.h"
#include <assert.h>

namespace tiny_ncnn{

/*
    左上角填充
*/
template<typename T>
static void copy_make_border_image(const Mat& src, Mat& dst, int top, int left, T v)
{
    int w = dst.w;
    int h = dst.h;

    const T* ptr = src;
    T* outptr = dst;

    int y = 0;
    // fill top
    for (; y < top; y++)
    {
        int x = 0;
        for (; x < w; x++)
        {
            outptr[x] = v;
        }
        outptr += w;
    }
    // fill center
    for (; y < (top + src.h); y++)
    {
        int x = 0;
        for (; x < left; x++)
        {
            outptr[x] = v;
        }
        if (src.w < 12)
        {
            for (; x < (left + src.w); x++)
            {
                outptr[x] = ptr[x - left];
            }
        }
        else
        {
            memcpy(outptr + left, ptr, src.w * sizeof(T));
            x += src.w;
        }
        for (; x < w; x++)
        {
            outptr[x] = v;
        }
        ptr += src.w;
        outptr += w;
    }
    // fill bottom
    for (; y < h; y++)
    {
        int x = 0;
        for (; x < w; x++)
        {
            outptr[x] = v;
        }
        outptr += w;
    }
    
}

void Padding::set_param(int _top, int _bottom, int _left, int _right, int _front, int _behind, float _value){
    top = _top;
    bottom = _bottom;
    left = _left;
    right = _right;
    front = _front;
    behind = _behind;
    value = _value;
}

/*
    Convolution:
        bottom_blobs 和 top_blobs 有且只有一个 不支持多输入多输出
        支持 1-dim 2-dim 3-dim 4-dim
*/
int Padding::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const{
    assert(bottom_blobs.size() == 1);
    assert(top_blobs.size() == 1);
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    Shape out_shape = bottom_blob.shape;
    size_t& w = std::get<0>(out_shape);
    size_t& h = std::get<1>(out_shape);
    size_t& d = std::get<2>(out_shape);
    size_t& c = std::get<3>(out_shape);
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    
    w += + left + right;
    if(dims == 1){
        top_blob = Mat(out_shape, dims, elemsize);
        copy_make_border_image<float>(bottom_blob, top_blob, 0, left, value);
        return 0;
    }

    h += + top + bottom;
    if(dims == 2){
        top_blob = Mat(out_shape, dims, elemsize);
        copy_make_border_image<float>(bottom_blob, top_blob, top, left, value);
        return 0;
    }

    if(dims == 3){
        int c_raw = c;
        c += front + behind;
        top_blob = Mat(out_shape, dims, elemsize);
        // Channel Padding
        for(int q=0; q<c; q++){
            Mat borderm = top_blob.channel(q);

            if((q < front) || (q >= (c_raw + front))){
                borderm.fill(value);
            }
            else{
                int q_ = q - front;
                const Mat m = bottom_blob.channel(q_);
                copy_make_border_image<float>(m, borderm, top, left, value);
            }
        }
        return 0;
    }

    if (dims == 4){
        int d_raw = d;
        d += front + behind;
        top_blob = Mat(out_shape, dims, elemsize);
        for(int q=0; q<d; q++){
            Mat borderm = top_blob.channel(q);

            if((q < front) || (q >= (d_raw + front))){
                borderm.fill(value);
            }
            else{
                int q_ = q - front;
                const Mat m = bottom_blob.channel(q_);
                copy_make_border_image<float>(m, borderm, top, left, value);
            }
        }
        return 0;
    }
    return 0;
}

Layer* Padding_layer_creator(void*){
    return new Padding();
}

}