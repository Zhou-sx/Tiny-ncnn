#ifndef NCNN_BLOB_H
#define NCNN_BLOB_H

#include <string>
using namespace std;

namespace tiny_ncnn{

class Blob{
public:
    Blob():input(-1) {};
    Blob(int _input): input(_input){}
    
    string name;      // 节点名称 方便索引
    int    input;     // 节点的输入层

    // int    node_id;   // 节点在图中的编号 可以不用 由在vector中的序号唯一标识
    // int    output;    // 节点的输出层
};

}

#endif