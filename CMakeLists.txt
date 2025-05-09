cmake_minimum_required(VERSION 3.8)
project(owl_vit_detector)
# Enable C++17
add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)

# === DEPENDENCIAS ===
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)  
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(OpenCV REQUIRED)

# External dependencies via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

# === INCLUDE DIRECTORIES ===
include_directories(
  include
  src
  ${GST_INCLUDE_DIRS}
  ${GST_BASE_INCLUDE_DIRS}
  ${GST_APP_INCLUDE_DIRS}
  ${GLIB_INCLUDE_DIRS}
  ${LIBUSB_INCLUDE_DIRS}
  ${rclcpp_components_INCLUDE_DIRS}
  /usr/local/include
)

# === LINK DIRECTORIES ===
link_directories(
  ${GST_LIBRARY_DIRS}
  ${GST_BASE_LIBRARY_DIRS}
  ${GST_APP_LIBRARY_DIRS}
  ${GLIB_LIBRARY_DIRS}
  ${LIBUSB_LIBRARY_DIRS}
  /usr/local/lib
)

# === C++ SHARED LIB ===
add_library(owl_vit_detector SHARED
  src/ricoh_theta_node.cpp
  src/thetauvc.c
)

target_link_libraries(owl_vit_detector
  ${OpenCV_LIBS}
  ${GST_LIBRARIES}
  ${GST_BASE_LIBRARIES}
  ${GST_APP_LIBRARIES}
  ${GLIB_LIBRARIES}
  ${LIBUSB_LIBRARIES}
  uvc
)

ament_target_dependencies(owl_vit_detector
  rclcpp
  sensor_msgs
  rclcpp_components
)

# === COMPONENT REGISTRATION ===
rclcpp_components_register_nodes(owl_vit_detector "theta_driver::ThetaDriver")

# === C++ EXECUTABLE ===
add_executable(theta_node src/theta_node.cpp)
target_link_libraries(theta_node owl_vit_detector)
ament_target_dependencies(theta_node rclcpp sensor_msgs rclcpp_components)

# === INSTALL LIBS AND BINARIES ===
install(TARGETS
owl_vit_detector
  theta_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)
install(FILES

  DESTINATION share/${PROJECT_NAME}
)


# === PYTHON CONFIGURACIÓN ===
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/owl_detector.py
  DESTINATION lib/${PROJECT_NAME}  
)


# Install launch files
#install(DIRECTORY launch
#  DESTINATION share/${PROJECT_NAME})

ament_package()