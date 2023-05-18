#ifndef NCNN_ALLOC_H
#define NCNN_ALLOC_H

#include <stdlib.h>

#define NCNN_MALLOC_ALIGN 64

namespace tiny_ncnn{

// sz上对齐至n
static size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

// 指针地址上对齐类型大小
template<typename _Tp>
static _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

class Alloc{
public:
    static void* allocate(size_t size);
    static void deallocate(void* ptr);
};


}

#endif