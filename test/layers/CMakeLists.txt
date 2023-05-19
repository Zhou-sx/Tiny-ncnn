macro(ncnn_add_layer_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ncnn)
endmacro()

# 包含src目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/layers)

ncnn_add_layer_test(absval_test)
ncnn_add_layer_test(padding_test)
ncnn_add_layer_test(conv_test)
ncnn_add_layer_test(innerproduct_test)
ncnn_add_layer_test(packing_test)