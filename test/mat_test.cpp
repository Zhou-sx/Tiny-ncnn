#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

    // 利用opencv读图像转换为mat
    cv::Mat img_bgr = cv::imread("/home/linaro/workspace/Tiny-ncnn/img/plane.jpg"), img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
    cv::Mat img_resize;
    cv::resize(img_rgb, img_resize, {227,227});
    
    if(!img_resize.isContinuous()){
        std::cout << "img is not Continuous." << std::endl;
        exit(-1);
    }
    Mat img = from_rgb_pixels(img_resize.data, 227, 227);
    // pretty_print(img.channel(0));

}