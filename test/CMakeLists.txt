macro(ncnn_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ncnn)
endmacro()

# 包含src目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

ncnn_add_test(alloc_test)
ncnn_add_test(mat_test)
ncnn_add_test(layer_test)
ncnn_add_test(net_test)
