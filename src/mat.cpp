#include <cstring>
#include <assert.h>
#include <stdlib.h>

#include "mat.h"
#include "platform.h"
#include "layer_factory.h"

namespace tiny_ncnn{

void Mat::check_shape(Shape& shape, int dims){
    auto& [w, h, d, c] = shape;
    // 根据dim修正shape
    switch (dims)
    {
    case 3:
        d = 1;
        break;
    case 2:
        d = 1;
        c = 1;
        break;
    case 1:
        d = 1;
        c = 1;
        h = 1;
        break;
    default:
        break;
    }
}

Mat Mat::clone() const{
    Mat m(shape, dims, elemsize, elempack);
    if(c_step == m.c_step){
        memcpy(m.data, data, total() * elemsize);
    }
    else{
        // 按照通道复制 两个Mat channel步长不一样
        size_t size = w * h * d * elemsize;
        for (int i = 0; i < c; i++)
        {
            memcpy(m.channel(i), channel(i), size);
        }
    }
    return m;
}

Mat::Mat(Shape _shape, size_t _dims, size_t _elemsize, size_t _elempack): 
    shape(_shape), data(nullptr), count(nullptr){
    check_shape(shape, _dims);
    allocator = new Alloc();
    create_buffer(_shape, _dims, _elemsize, _elempack);
}

Mat::Mat(const Mat& rhs):shape(rhs.shape), c_step(rhs.c_step), dims(rhs.dims),elemsize(rhs.elemsize),
elempack(rhs.elempack), allocator(rhs.allocator), data(rhs.data), count(rhs.count){
    w = rhs.w;
    if(count != nullptr){
        *count += 1;
    }
}

Mat::Mat(Shape _shape, size_t _dims, void* _data, size_t _elemsize, size_t _elempack, Alloc* p_alloc)
    : shape(_shape), dims(_dims), elemsize(_elemsize), elempack(_elempack), allocator(p_alloc), data(_data), count(nullptr)
{
    check_shape(shape, _dims);
    const auto [_w, _h, _d, _c] = shape;
    // 每个channel pad 对齐至16字节
    switch (_dims)
    {
    case 4:
        c_step = alignSize(_w*_h*_d*_elemsize, 16) / _elemsize * _elempack;
        break;
    case 3:
        c_step = alignSize(_w*_h*_elemsize, 16) / _elemsize * _elempack;
        break;
    case 2:
        c_step = _w * _h * _elempack;
        break;
    case 1:
        c_step = _w * _elempack;
        break;
    }
}

Mat& Mat::operator=(const Mat& rhs){
    if(this == &rhs){
        return *this;
    }
    release();

    w = rhs.w;
    h = rhs.h;
    c = rhs.c;
    d = rhs.d;
    c_step = rhs.c_step;
    dims = rhs.dims;
    elemsize = rhs.elemsize;
    elempack = rhs.elempack;
    allocator = rhs.allocator;
    data = rhs.data;
    count = rhs.count;

    if(count != nullptr){
        *count += 1;
    }
    return *this;
}

/*
    _elemsize: 是pack后的大小 不需要再乘elempack
*/
void Mat::create_buffer(Shape _shape, size_t _dims, size_t _elemsize, size_t _elempack){
    // c++ 17
    check_shape(_shape, _dims);
    const auto [_w, _h, _d, _c] = _shape;
    // 如果和现在的buffer一样则不需要再开辟空间
    if (dims == _dims && w == _w && h == _h && c == _c && d == _d && elemsize == _elemsize)
        return;
    release();

    elemsize = _elemsize;
    elempack = _elempack;
    dims = _dims;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    // 每个channel pad 对齐至16字节
    switch (_dims)
    {
    case 4:
        c_step = alignSize(_w*_h*_d*_elemsize, 16) / _elemsize * _elempack;
        break;
    case 3:
        c_step = alignSize(_w*_h*_elemsize, 16) / _elemsize * _elempack;
        break;
    case 2:
        c_step = _w * _h * _elempack;
        break;
    case 1:
        c_step = _w * _elempack;
        break;
    }
    

    if(total() > 0){
        // 总体的内存 对齐至4字节(*count大小) 再补一个count的位置
        size_t totalsize = alignSize(total() * _elemsize, sizeof(*count)) + sizeof(*count);
        data = allocator->allocate(totalsize);
        
        // 引用数为1
        // 先按照字节数找到count的位置，然后把指针类型强制转换成size_t*，就可以写了
        count = reinterpret_cast<size_t*>(&static_cast<unsigned char *>(data)[total() * _elemsize]);
        *count = 1;
    }
}

void Mat::release(){
    if(count != nullptr && --(*count) == 0){
        allocator->deallocate(data);
    }
    data = nullptr;

    elemsize = 0;

    dims = 0;
    w = 0;
    h = 0;
    d = 0;
    c = 0;

    c_step = 0;

    count = nullptr;
}

/*
    先利用基于内存地址的构造函数生成一个初始化 Mat
    除了4->3 其他channel操作和构造出来的Mat一样 因为 3->2 2->1 都没有channel padding
*/

Mat Mat::channel(int c) const{
    void* ptr =  static_cast<void*> (&static_cast<unsigned char*>(data)[c_step * elemsize / elempack * c]);
    int _d = d; 
    Mat m(shape, dims-1, ptr, elemsize, elempack, allocator);
    if(dims == 4){
        m.c_step = (size_t)w * h * elempack; // 在channel有padding 降一维 没有padding
        m.c = _d;                            // 4降3需要把 d 和 c 作交换
    }
    return m;
}


Mat Mat::reshape(Shape _shape, size_t _dims) const{
    // assert(elempack == 1);
    check_shape(_shape, _dims);
    const auto [_w, _h, _d, _c] = _shape;
    // 尺寸不匹配
    if (w * h * d * c != _w * _h * _d * _c)
        return Mat();

    // 此时转换后没有channel间没有padding
    if(_dims <= 2){
        // 转换前channel间有padding 先Flatten去掉padding
        if(dims >=3 && c_step != w*h*d){
            Mat m(_shape, _dims, elemsize, elempack);
            // 每个通道拷贝
            for (int i = 0; i < c; i++)
            {
                const void* ptr = (unsigned char*)data + i * c_step * elemsize;
                void* mptr = (unsigned char*)m.data + i * w * h * d * elemsize;
                memcpy(mptr, ptr, w * h * d * elemsize);
            }
            return m;
        }
        // 如果没有padding 无需flatten 只要改变c_step即可
        Mat m = *this;

        m.dims = _dims;
        m.w = _w;
        m.h = _h;
        m.d = _d;
        m.c = _c;

        switch (_dims)
        {
        case 2:
            m.c_step = _w * _h * elempack;
            break;
        case 1:
            m.c_step = _w * elempack;
            break;
        }
        return m;
    }
    else{
        // 转换后有padding 

        // 和上面一样先判断转换前有没有padding
        if(dims >=3 && c_step != w*h*d){
            // 先转换到一个没有padding的
            Mat m = this->reshape({_w * _h * _d * _c, 1, 1, 1}, 1);
            // 转换前没有padding 递归调用
            return m.reshape(_shape, _dims);
        }
        // 转换后有padding 转换前没有padding
        Mat m(_shape, _dims, elemsize, elempack);
        // 每个通道拷贝
        for (int i = 0; i < _c; i++)
        {
            const void* ptr = (unsigned char*)data + i * _w * _h * _d * elemsize;
            void* mptr = (unsigned char*)m.data + i * m.c_step * elemsize;
            memcpy(mptr, ptr, _w * _h * _d * elemsize);
        }
        return m;
    }
}

/*
    归一化
*/
void Mat::substract_mean_normalize(const float* mean_vals, const float* norm_vals){
    for (int i = 0; i < c; i++){
        float mean = mean_vals[i];
        float norm = norm_vals[i];
        Mat m_i = this->channel(i);
        float* data_i = static_cast<float*>(m_i.data);

        for(int j = 0; j < c_step; j++){
            *data_i -=  mean;
            *data_i *= norm;
            data_i++;
        }
    }
}

/*
    复用 Padding Layer 给Mat四周填充
*/
void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, float v)
{
    Layer* padding = create_layer(3);

    dynamic_cast<Padding*>(padding)->set_param(top, bottom, left, right, 0, 0, v);

    std::vector<Mat> input = {src};
    std::vector<Mat> output = {dst};
    padding->forward(input, output);
    dst = output[0];

    delete padding;
}


/*
    用opencv中的Mat 生成 tiny_ncnn中的Mat
*/
Mat from_rgb_pixels(const unsigned char* pixels, int w, int h){
    Mat m({w, h, 1, 3}, 3, 4);

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++){
            *ptr0 = pixels[0];
            *ptr1 = pixels[1];
            *ptr2 = pixels[2];

            pixels += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }
    }
    return m;
}

/*
    生成随机Mat
*/
float RandomFloat(float a, float b){
    float random = ((float)rand()) / (float)RAND_MAX; //RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    float v = a + r;
    // generate denormal as zero
    if (v < 0.0001 && v > -0.0001)
        v = 0.f;
    return v;
}

void Randomize(Mat& m, float a, float b)
{   
    for (size_t i = 0; i < m.total(); i++)
    {
        m[i] = RandomFloat(a, b);
    }
}

Mat RandomMat(Shape _shape, size_t _dims, float a, float b)
{
    Mat m(_shape, _dims, 4, 1);
    Randomize(m, a, b);
    return m;
}

}