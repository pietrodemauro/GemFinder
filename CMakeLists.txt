cmake_minimum_required(VERSION 3.20)

project(GemFinder
    VERSION 0.1.0
    DESCRIPTION "Desktop Multimodal Semantic Search Engine in C++20"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization and Warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wno-unused-parameter
    )
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O3 -march=native)
    endif()
endif()

# Find external dependencies
find_package(SQLite3 REQUIRED)

find_path(XXHASH_INCLUDE_DIR NAMES xxhash.h)
find_library(XXHASH_LIBRARY NAMES xxhash)

if(NOT XXHASH_INCLUDE_DIR OR NOT XXHASH_LIBRARY)
    message(FATAL_ERROR "xxhash not found. Please install xxhash development package.")
endif()

# llama.cpp subproject configuration
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# Common include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/usearch/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/miniaudio
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${SQLite3_INCLUDE_DIRS}
    ${XXHASH_INCLUDE_DIR}
)

# gemfinder_common interface library
add_library(gemfinder_common INTERFACE)
target_include_directories(gemfinder_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/usearch/include
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/miniaudio
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    SYSTEM ${SQLite3_INCLUDE_DIRS}
    SYSTEM ${XXHASH_INCLUDE_DIR}
)
target_link_libraries(gemfinder_common INTERFACE
    ${SQLite3_LIBRARIES}
    ${XXHASH_LIBRARY}
    m
)

# gemfinder_storage library
add_library(gemfinder_storage STATIC
    src/storage/Database.cpp
    src/storage/VectorStore.cpp
    src/storage/StorageManager.cpp
)
target_link_libraries(gemfinder_storage PUBLIC
    gemfinder_common
)

# gemfinder_preprocessor library
add_library(gemfinder_preprocessor STATIC
    src/preprocessor/TextPreprocessor.cpp
    src/preprocessor/ImagePreprocessor.cpp
    src/preprocessor/AudioPreprocessor.cpp
    src/preprocessor/MediaPreprocessor.cpp
)
target_link_libraries(gemfinder_preprocessor PUBLIC
    gemfinder_common
)

# gemfinder_inference library
add_library(gemfinder_inference STATIC
    src/inference/MockEngine.cpp
    src/inference/LlamaEngine.cpp
    src/inference/EngineFactory.cpp
)
target_link_libraries(gemfinder_inference PUBLIC
    gemfinder_common
    llama
    ggml
)

# CLI executable
add_executable(gemfinder-cli
    src/cli/main.cpp
)
target_link_libraries(gemfinder-cli PRIVATE
    gemfinder_storage
    gemfinder_preprocessor
    gemfinder_inference
)

# Tests
enable_testing()

add_executable(test_storage
    tests/test_storage.cpp
)
target_link_libraries(test_storage PRIVATE
    gemfinder_storage
)

add_executable(test_preprocessor
    tests/test_preprocessor.cpp
)
target_link_libraries(test_preprocessor PRIVATE
    gemfinder_preprocessor
)

add_executable(test_inference
    tests/test_inference.cpp
)
target_link_libraries(test_inference PRIVATE
    gemfinder_inference
)

add_test(NAME StorageLayerTest COMMAND test_storage)
add_test(NAME PreprocessorTest COMMAND test_preprocessor)
add_test(NAME InferenceEngineTest COMMAND test_inference)
