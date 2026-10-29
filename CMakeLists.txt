cmake_minimum_required(VERSION 3.20)
project(mnist CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(mnist
  src/main.cpp
  src/mnist_dataset.cpp
  src/matmul.cpp
  src/network.cpp
)

target_compile_options(mnist PRIVATE -Wall -Wextra)

# Debug builds run with AddressSanitizer + UBSan to catch out-of-bounds bugs.
target_compile_options(mnist PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
target_link_options(mnist PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
