macro(ncnn_add_layer class)
    list(APPEND ncnn_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/layers/${class}.cpp)
endmacro()

set(ncnn_SRCS
    net.cpp
    alloc.cpp
    mat.cpp
    extractor.cpp
    layer.cpp
    data_reader.cpp
    paramdict.cpp
    modelbin.cpp
    layer_factory.cpp
)

ncnn_add_layer(absval)
ncnn_add_layer(concat)
ncnn_add_layer(split)
ncnn_add_layer(padding)
ncnn_add_layer(packing)
ncnn_add_layer(conv)
ncnn_add_layer(input)
ncnn_add_layer(relu)
ncnn_add_layer(lrn)
ncnn_add_layer(pooling)
ncnn_add_layer(dropout)
ncnn_add_layer(softmax)
ncnn_add_layer(innerproduct)
ncnn_add_layer(convolutiondepthwise)

add_library(ncnn SHARED ${ncnn_SRCS})