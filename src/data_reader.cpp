#include <vector>
#include <sstream>

#include "data_reader.h"
#include "platform.h"

namespace tiny_ncnn{

int DataReader::scan(const char* format, void* p) const{
    return fscanf(fp, format, p);
}

int DataReader::read(void* buf, size_t size) const{
    return fread(buf, 1, size, fp);
}

}