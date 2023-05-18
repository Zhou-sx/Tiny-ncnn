#include <cassert>
#include <cstdint>
#include "alloc.h"

namespace tiny_ncnn{

/*
    申请size字节的内存 返回一个对齐的首地址
    ncnn中的手动实现方法是，在数据段前一个位置设置一个指针，指向malloc分配的不对齐地址
    这个方法可以保证allocate返回一个对齐的地址同时可以用原指针释放空间
    
    在重写的过程中，发现在c++17起，new已经支持了内存对齐(ncnn仓库里也有多种方式)
    编译器默认16字节对齐
*/

void* Alloc::allocate(size_t size){
    #if __cplusplus >= 201703L 
        return ::operator new(size, std::align_val_t{NCNN_MALLOC_ALIGN});
    #else
        u_char* ptr = (u_char *)::operator new(size + sizeof(void*) + NCNN_MALLOC_ALIGN);
        u_char** data = (u_char **)alignPtr((u_char *)ptr, NCNN_MALLOC_ALIGN); 
        assert(reinterpret_cast<uintptr_t>(data) % NCNN_MALLOC_ALIGN == 0);

        // 为什么要用指向指针的指针呢 方便数组索引 因为data[-1]放的是一个指针
        data[-1] = ptr;
        return static_cast<void*>(data);
    #endif
        
}

void Alloc::deallocate(void* ptr){
    #if __cplusplus >= 201703L 
        ::operator delete(ptr, std::align_val_t{NCNN_MALLOC_ALIGN});
    #else  
        u_char* raw_ptr = static_cast<u_char **>(ptr)[-1];
        ::operator delete(raw_ptr);
    #endif  
    return;
}

}