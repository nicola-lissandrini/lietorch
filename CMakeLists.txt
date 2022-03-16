cmake_minimum_required(VERSION 3.0.2)
project(lietorch)

add_compile_options(-std=c++14)

set (Torch_DIR "/opt/libtorch/share/cmake/Torch")

find_package (Torch REQUIRED)

include_directories (
	include
	${TORCH_INCLUDE_DIRS}
)

add_library (lietorch SHARED
	src/quaternion.cpp
	src/pose.cpp)

set_target_properties (
	lietorch PROPERTIES
	VERSION 0.9
	SOVERSION 0.9 )

# Install
install (TARGETS lietorch
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/lietorch DESTINATION include)
export (PACKAGE lietorch)
