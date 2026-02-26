cmake_minimum_required(VERSION 3.20)
project(vector_add LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Download metal-cpp if not present
set(METAL_CPP_DIR "${CMAKE_SOURCE_DIR}/metal-cpp/metal-cpp")
if(NOT EXISTS "${METAL_CPP_DIR}")
    message(STATUS "Downloading metal-cpp...")
    file(DOWNLOAD
        "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
        "${CMAKE_BINARY_DIR}/metal-cpp.zip"
    )
    file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/metal-cpp.zip"
         DESTINATION "${CMAKE_SOURCE_DIR}/metal-cpp"
    )
endif()

# Build the host executable
add_executable(vector_add main.cpp verify_vector_add.cpp)
target_include_directories(vector_add PRIVATE ${METAL_CPP_DIR})
target_link_libraries(vector_add PRIVATE
    "-framework Metal"
    "-framework Foundation"
)

# Copy the Metal shader to the build directory
configure_file(vector_add.metal ${CMAKE_BINARY_DIR}/vector_add.metal COPYONLY)
