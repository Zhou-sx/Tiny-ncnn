#include "modelbin.h"
#include "platform.h"

namespace tiny_ncnn{

Mat ModelBin::load(int w, int type) const{
    Mat m;

    if (type == 0){
        size_t nread;

        // 揭示权重信息
        union
        {
            struct
            {
                unsigned char f0;
                unsigned char f1;
                unsigned char f2;
                unsigned char f3;
            };
            unsigned int tag;
        } flag_struct;
        nread = dr.read(&flag_struct, sizeof(flag_struct));

        unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

        /*
        flag : unsigned int, little-endian, indicating the weight storage type, 
             0          => float32, 
             0x01306B47 => float16, 
             0x000D4B38 => int8, 
             0x0002C056 => raw data with extra scaling  带有尺度信息的 float32
	         其他 非0   =>  quantized data  256个量化数 和 索引表
        */
        // try reference data
        if(flag_struct.tag == 0x01306B47){
            // skip
        }
        else if(flag_struct.tag == 0x000D4B38){
            // skip
        }
        else if(flag_struct.tag == 0x0002C056){
            // skip
        }

        if(flag != 0){

        }
        else if(flag_struct.f0 == 0)
        {
            m = Mat(Shape(w, 1, 1, 1), 1, 4);
            if (m.empty())
                return m;

            // raw data
            size_t nread = dr.read(m, w * sizeof(float));
            if (nread != w * sizeof(float))
            {
                NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
                return Mat();
            }
            
            return m;
        }


    }
    else if (type == 1){
        m = Mat(Shape(w, 1, 1, 1), 1, 4);
        if (m.empty())
            return m;

        // raw data
        size_t nread = dr.read(m, w * sizeof(float));
        if (nread != w * sizeof(float))
        {
            NCNN_LOGE("ModelBin read weight_data failed %zd", nread);
            return Mat();
        }
        
        return m;
    }

    return m;
}

}