// src/world_manager_main.cpp
#include <rclcpp/rclcpp.hpp>
#include "robot_world/world_manager.hpp" // 包含头文件

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WorldManager>());
  rclcpp::shutdown();
  return 0;
}
