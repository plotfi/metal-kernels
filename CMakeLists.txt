cmake_minimum_required(VERSION 3.20)
project(metal_kernel LANGUAGES CXX)

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
add_executable(metal_kernel main.cpp)
target_include_directories(metal_kernel PRIVATE ${METAL_CPP_DIR})
target_link_libraries(metal_kernel PRIVATE
    "-framework Metal"
    "-framework Foundation"
)

# Copy all Metal shaders to the build directory
file(GLOB METAL_SHADERS "${CMAKE_SOURCE_DIR}/*.metal")
foreach(SHADER ${METAL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    configure_file(${SHADER} ${CMAKE_BINARY_DIR}/${SHADER_NAME} COPYONLY)
endforeach()
