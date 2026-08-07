file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()

foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_all_modules_sources(OPTYPE sorted_nms ACLNNTYPE aclnn_exclude
                        COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
                        TILING_DIR ${SUPPORT_TILING_DIR})
