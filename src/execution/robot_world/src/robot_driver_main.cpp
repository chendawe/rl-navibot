// src/robot_driver_main.cpp
#include <rclcpp/rclcpp.hpp>
#include "robot_world/robot_driver.hpp" // 包含头文件

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // 实例化类并 spin
  rclcpp::spin(std::make_shared<RobotDriver>());
  rclcpp::shutdown();
  return 0;
}
