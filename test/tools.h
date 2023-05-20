#include "../src/mat.h"
#include <cstdio>

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