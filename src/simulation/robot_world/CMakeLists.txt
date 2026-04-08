cmake_minimum_required(VERSION 3.8)
project(robot_world)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# === 1. 补全缺失的依赖包 ===
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)       # robot_driver 用到了 Float32MultiArray
find_package(tf2 REQUIRED)            # robot_driver 用到了 tf2/utils.h
find_package(tf2_geometry_msgs REQUIRED) # robot_driver 用到了 tf2 转换
find_package(gazebo_msgs REQUIRED)    # world_manager 用到了
find_package(std_srvs REQUIRED)       # world_manager 用到了 Empty.srv

# === 2. 推荐：将核心代码编译为库 (为了方便测试) ===
# 这样 test 文件可以直接链接这个库，而不需要重新编译源文件
add_library(${PROJECT_NAME}_lib
  src/robot_driver.cpp
  src/world_manager.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  gazebo_msgs
  std_srvs
)

# === 3. 编译可执行文件 (节点) ===
# 注意：如果你的源文件里包含了 main 函数，这里直接链接库会导致 "multiple main" 错误。
# 最佳实践是将 main 函数单独放在一个 cpp 里，或者像下面这样单独编译 exe。

# 方案 A (推荐)：单独编译 exe (如果不想拆分 main 函数)
add_executable(robot_driver src/robot_driver_main.cpp)
target_link_libraries(robot_driver ${PROJECT_NAME}_lib)
ament_target_dependencies(robot_driver rclcpp)

add_executable(world_manager src/world_manager_main.cpp)
target_link_libraries(world_manager ${PROJECT_NAME}_lib)
ament_target_dependencies(world_manager rclcpp)

# === 4. 安装 ===
install(TARGETS
  robot_driver
  world_manager
  DESTINATION lib/${PROJECT_NAME}
)

# 安装 Python Launch 文件
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# === 5. 测试配置 ===
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # --- 测试 RobotDriver ---
  ament_add_gtest(test_robot_driver test/test_robot_driver.cpp)
  # 关键：给测试文件链接依赖
  target_link_libraries(test_robot_driver ${PROJECT_NAME}_lib) # 链接库
  ament_target_dependencies(test_robot_driver rclcpp geometry_msgs sensor_msgs nav_msgs std_msgs tf2 tf2_geometry_msgs)
  # 如果你的 test_robot_driver.cpp 里 #include 了源码，就不需要 target_link_libraries
  # 如果你拆分了头文件，就需要链接库： target_link_libraries(test_robot_driver ${PROJECT_NAME}_lib)

  # --- 测试 WorldManager ---
  ament_add_gtest(test_world_manager test/test_world_manager.cpp)
  target_link_libraries(test_world_manager ${PROJECT_NAME}_lib) # 链接库
  ament_target_dependencies(test_world_manager rclcpp gazebo_msgs std_srvs)

endif()

ament_package()
