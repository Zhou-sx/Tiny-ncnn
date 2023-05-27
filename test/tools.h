#include "../src/mat.h"
#include "../src/layer.h"
#include <cstdio>
#include <sys/time.h>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdlib.h>

namespace tiny_ncnn{

void pretty_print(const tiny_ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w * m.elempack;)
            {
                if(m.elempack == 1){
                    printf("%.2f ", ptr[x++]);
                }
                else{
                    printf("(");
                    for(int k=0; k<m.elempack; k++){
                        printf("%.2f ", ptr[x++]);
                    }
                    printf(") ");
                }
            }
            ptr += m.w * m.elempack;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

static double gtod_ref_time_sec = 0.0;

/* Adapted from the bl2_clock() routine in the BLIS library */

double dclock()
{
        double the_time, norm_sec;
        struct timeval tv;

        gettimeofday( &tv, NULL );

        if ( gtod_ref_time_sec == 0.0 )
                gtod_ref_time_sec = ( double ) tv.tv_sec;

        norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;

        the_time = norm_sec + tv.tv_usec * 1.0e-6;

        return the_time;
}

static double get_time(struct timespec *start,
                       struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

static int set_sched_affinity(size_t thread_affinity_mask)
{
    pid_t pid = syscall(SYS_gettid);

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(thread_affinity_mask), &thread_affinity_mask);
    if (syscallret)
    {
        fprintf(stderr, "syscall error %d\n", syscallret);
        return -1;
    }
    return 0;
}

void time_test(Layer* p, Mat& a, double gflops){
    double time_tmp, time_best;
    struct timespec start, end;
    double time_used = 0.0;

    // 绑定CPU测试
    size_t mask = 0;
    mask |= (1 << 0);
    set_sched_affinity(mask);


    Mat c;
    std::vector<Mat> input = {a};
    std::vector<Mat> output = {c};

    // 循环20次，以最快的运行时间为结果
    for(int j=0; j < 20; j++){
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        
        // do forward
        p->forward(input, output);

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_tmp = get_time(&start, &end);
        
        if(j == 0){
            time_best = time_tmp;
        }
        else{
            time_best = std::min(time_best, time_tmp);
        }
            
    }
    printf("%le\n", gflops / time_best);

}
}
