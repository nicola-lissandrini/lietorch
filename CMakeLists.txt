cmake_minimum_required(VERSION 3.0.2)
project(lietorch)

add_compile_options(-std=c++14)

set (Torch_DIR "/opt/libtorch/share/cmake/Torch")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package (Torch REQUIRED)

include_directories (
	include
	${TORCH_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)

add_library (lietorch SHARED
	src/quaternion.cpp
	src/pose.cpp)

set_target_properties (
	lietorch PROPERTIES
	VERSION 0.9.1
	SOVERSION 0.9.1 )

add_executable(test src/test.cpp)
target_link_libraries (test ${TORCH_LIBRARIES} lietorch)

# Install
install (TARGETS lietorch
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/lietorch DESTINATION include)
export (PACKAGE lietorch)
