#include "alloc.h"

int main(){
    tiny_ncnn::Alloc alloc{};
    
    void* p = alloc.allocate(1025);
    alloc.deallocate(p);
    
    return 0;
}