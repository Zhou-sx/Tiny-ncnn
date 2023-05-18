#include <iostream>

#include "mat.h"
#include "tools.h"
using namespace tiny_ncnn;

int main(){
    Mat m({4, 4, 1, 3}, 3, 4);
    Mat m2{};

    Mat m3(m);
    m2 = m;

    Mat m4 = m.channel(0);
    m4.fill(1.0f);
    std::cout << "m4 value: " << std::endl;
    pretty_print(m4);

    Mat m5 = m4.channel(0);
    m5.fill(1.5f);
    std::cout << "m5 value: " << std::endl;
    pretty_print(m5);

    std::cout << "m value: " << std::endl;
    pretty_print(m);

    Mat m6 = m.clone();
    std::cout << "m6 count: " << *(m6.count) << std::endl;
    std::cout << "m6 value: " << std::endl;
    pretty_print(m6);

    Mat m7 = m.reshape({4, 12, 1, 1}, 2);
    std::cout << "m7 value: " << std::endl;
    pretty_print(m7);

    Mat m8 = m7.reshape({6, 8, 1, 1}, 2);
    std::cout << "m8 value: " << std::endl;
    pretty_print(m8);


    Mat m9 = m7.reshape({4, 2, 2, 3}, 4);
    Mat m10 = m9.channel(0);
    std::cout << "m10 value: " << std::endl;
    pretty_print(m10);

    std::vector<Mat> v = {m, m2, m3};
    const std::vector<Mat>& v2 = v;

}