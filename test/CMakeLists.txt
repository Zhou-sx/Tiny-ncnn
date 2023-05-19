macro(ncnn_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ncnn)
endmacro()

# 包含src目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(
    include_libs  
    /home/linaro/workspace/opencv/include/opencv4
)
include_directories(${include_libs})

# 添加动态链接库opencv及pthread
set(opencv_lib "/home/linaro/workspace/opencv/lib")
set(
    dynamic_libs   
    ${opencv_lib}/libopencv_core.so
    ${opencv_lib}/libopencv_highgui.so
    ${opencv_lib}/libopencv_imgcodecs.so
    ${opencv_lib}/libopencv_imgproc.so
)

ncnn_add_test(alloc_test)

ncnn_add_test(mat_test)
target_link_libraries(mat_test PRIVATE ${dynamic_libs})

ncnn_add_test(layer_test)

ncnn_add_test(net_test)
target_link_libraries(net_test PRIVATE ${dynamic_libs})
