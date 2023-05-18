#ifndef NCNN_PLATFORM_H
#define NCNN_PLATFORM_H

#include <stdio.h>
#define NCNN_LOGE(...) do { \
    fprintf(stderr, ##__VA_ARGS__); fprintf(stderr, "\n"); } while(0)


#endif