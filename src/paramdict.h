#ifndef NCNN_PARAMDICT_H
#define NCNN_PARAMDICT_H
#include "mat.h"
#include "data_reader.h"

const int NCNN_MAX_PARAM_COUNT = 128;

namespace tiny_ncnn{

class ParamDict{
public:
    ParamDict() = default;
    int load_param(const DataReader& dr);

    // get 
    int type(int id) const;
    int get(int id, int def) const;
    float get(int id, float def) const;

    // set
    void set(int id, int i);
    void set(int id, float i);

    void clear();

private:
    struct{
        // 0 = null
        // 1 = int
        // 2 = float
        int type;
        union
        {
            int i;
            float f;
        };
    } params[NCNN_MAX_PARAM_COUNT];

};

}

#endif