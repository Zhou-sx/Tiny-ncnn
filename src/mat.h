#ifndef NCNN_MAT_H
#define NCNN_MAT_H

#include <vector>
#include <tuple>
#include "alloc.h"

namespace tiny_ncnn{

// Shape: [w, h, d, c]
typedef std::tuple<size_t, size_t, size_t, size_t> Shape;


class Mat{
public:
    // 三五构造 使用浅拷贝
    Mat(): data(nullptr), allocator(nullptr), count(nullptr) {};
    Mat(const Mat& rhs);
    Mat& operator=(const Mat& rhs);
    // Mat(Mat&& rhs);
    // Mat operator=(Mat && rhs);
    
    Mat(Shape _shape, size_t _dims, size_t _elemsize, size_t _elempack = 1);
    
    /*
        根据已有的Mat取其部分构造新的Mat
        子部分的生命周期完全取决与大的 
        所以 count = nullptr
    */
    Mat(Shape _shape, size_t _dims, void* _data, size_t _elemsize, Alloc* p_alloc = nullptr);
    
    ~Mat() { release();}

    // 深拷贝
    Mat clone() const;

public:

    static void check_shape(Shape& shape, int dims);

    // 不同类型的填充 加以不同指令集的优化
    void fill(float v){
        int size = (int)total();
        float* ptr = (float*)data;

        for(int i=0; i<size; i++){
            ptr[i] = v;
        }
    }

    // 取某通道
    Mat channel(int c) const;

    // 强制转换
    template<typename T>
    operator T*() const{
        return static_cast<T*>(data);
    }

    // 只能用在float 谨慎使用
    float& operator[](size_t i){
        return (static_cast<float*>(data))[i];
    }

    const float& operator[](size_t i) const{
        return (static_cast<const float*>(data))[i];
    }

    // 取第y行
    template<typename T>
    T* row(int y){
        return (T*)((unsigned char*)data + (size_t)w * y * elemsize);
    }

    template<typename T>
    const T* row(int y) const{
        return (const T*)((unsigned char*)data + (size_t)w * y * elemsize);
    }

    // reshape 浅拷贝
    Mat reshape(Shape _shape, size_t _dims);

public:
    // 开辟长度为n的一段内存
    void create_buffer(Shape _shape, size_t _dims, size_t _elemsize);
    // 释放内存
    void release();

public:
    bool empty() const
    {
        return data == 0 || total() == 0;
    }

    size_t total() const
    {
        return c_step * c;
    }

public:
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

public:
    // 数据指针
    void* data;

    // alloc
    Alloc* allocator; 

    // count 实现智能指针功能
    size_t* count;
    
    // elemsize 元素大小
    size_t elemsize;
    // elempack pack个数
    size_t elempack;

    // 形状
    Shape shape;
    size_t& w = std::get<0>(shape);
    size_t& h = std::get<1>(shape);
    size_t& d = std::get<2>(shape);
    size_t& c = std::get<3>(shape);
    size_t dims;

    // channel跨步
    size_t c_step;

};

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, float v);

Mat from_rgb_pixels(const unsigned char* pixels, int w, int h);



}


#endif