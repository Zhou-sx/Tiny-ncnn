#ifndef NCNN_DATA_READER_H
#define NCNN_DATA_READER_H
#include <iostream>
#include <fstream>
#include <string>

namespace tiny_ncnn{

class DataReader{
public:
    DataReader():fp(nullptr) {};
    DataReader(FILE* _fp):fp(_fp) {};

    // 读参数
    int scan(const char* format, void* p) const;
    // 读二进制
    int read(void* buf, size_t size) const;

private:
    FILE* fp;
};

}

#endif