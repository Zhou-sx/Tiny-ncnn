#include <cstring>

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
    Mat m(shape, dims, elemsize);
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

Mat::Mat(Shape _shape, size_t _dims, size_t _elemsize, Alloc* p_alloc): 
    shape(_shape), elemsize(_elemsize), allocator(p_alloc), data(nullptr), count(nullptr){
    check_shape(shape, _dims);
    if(p_alloc == nullptr){
        allocator = new Alloc();
    }
    create_buffer(_shape, _dims, elemsize);
}

Mat::Mat(const Mat& rhs):shape(rhs.shape), c_step(rhs.c_step), dims(rhs.dims),
elemsize(rhs.elemsize), allocator(rhs.allocator), data(rhs.data), count(rhs.count){
    w = rhs.w;
    if(count != nullptr){
        *count += 1;
    }
}

Mat::Mat(Shape _shape, size_t _dims, void* _data, size_t _elemsize, Alloc* p_alloc)
    : shape(_shape), dims(_dims), elemsize(_elemsize), allocator(p_alloc), data(_data), count(nullptr)
{
    check_shape(shape, _dims);
    const auto [_w, _h, _d, _c] = shape;
    // 每个channel pad 对齐至16字节
    switch (_dims)
    {
    case 4:
        c_step = alignSize(_w*_h*_d*_elemsize, 16) / _elemsize;
        break;
    case 3:
        c_step = alignSize(_w*_h*_elemsize, 16) / _elemsize;
        break;
    case 2:
        c_step = _w * _h;
        break;
    case 1:
        c_step = _w;
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
    allocator = rhs.allocator;
    data = rhs.data;
    count = rhs.count;

    if(count != nullptr){
        *count += 1;
    }
    return *this;
}

void Mat::create_buffer(Shape _shape, size_t _dims, size_t _elemsize){
    // c++ 17
    check_shape(_shape, _dims);
    const auto [_w, _h, _d, _c] = _shape;
    // 如果和现在的buffer一样则不需要再开辟空间
    if (dims == _dims && w == _w && h == _h && c == _c && d == _d && elemsize == _elemsize)
        return;
    release();

    elemsize = _elemsize;
    dims = _dims;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    // 每个channel pad 对齐至16字节
    switch (_dims)
    {
    case 4:
        c_step = alignSize(_w*_h*_d*_elemsize, 16) / _elemsize;
        break;
    case 3:
        c_step = alignSize(_w*_h*_elemsize, 16) / _elemsize;
        break;
    case 2:
        c_step = _w * _h;
        break;
    case 1:
        c_step = _w;
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
    void* ptr =  static_cast<void*> (&static_cast<unsigned char*>(data)[c_step * elemsize * c]);
    int _d = d; 
    Mat m(shape, dims-1, ptr, elemsize, allocator);
    if(dims == 4){
        m.c_step = (size_t)w * h; // 在channel有padding 降一维 没有padding
        m.c = _d;                 // 4降3需要把 d 和 c 作交换
    }
    return m;
}


Mat Mat::reshape(Shape _shape, size_t _dims){
    check_shape(_shape, _dims);
    const auto [_w, _h, _d, _c] = _shape;
    // 尺寸不匹配
    if (w * h * d * c != _w * _h * _d * _c)
        return Mat();

    // 此时转换后没有channel间没有padding
    if(_dims <= 2){
        // 转换前channel间有padding 先Flatten去掉padding
        if(dims >=3 && c_step != w*h*d){
            Mat m(_shape, _dims, elemsize);
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
            c_step = _w * _h;
            break;
        case 1:
            c_step = _w;
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
        Mat m(_shape, _dims, elemsize);
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


}