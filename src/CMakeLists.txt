file(GLOB VC_LIBRARY_SOURCES ${VC_SOURCE_DIR}/src/*.cpp)
file(GLOB VC_LIBRARY_HEADERS ${VC_SOURCE_DIR}/src/*.h)
file(GLOB VC_LIBRARY_CUDA_HEADERS ${VC_SOURCE_DIR}/src/*.cuh)

cuda_add_library(vc_lib ${VC_LIBRARY_SOURCES})

include_directories(${CUDA_INCLUDE_DIR} ${VC_SOURCE_DIR}/src/ )

cuda_add_executable(vc ${VC_SOURCE_DIR}/src/main.cu ${VC_LIBRARY_HEADERS} ${VC_LIBRARY_CUDA_HEADERS})
set_target_properties(vc PROPERTIES
CUDA_ARCHITECTURES "35;37;60;70;80")
target_compile_options(vc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
#target_compile_options(match PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_37,code=sm_37>)
target_link_libraries(vc ${CUDA_LIBRARY} cuda vc_lib)