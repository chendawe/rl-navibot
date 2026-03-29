// include/robot_world/robot_driver.hpp
#pragma once  // 告诉编译器：这个文件只编译一次，简单粗暴
// #ifndef ROBOT_WORLD_ROBOT_DRIVER_HPP  // 如果没有定义这个宏
// #define ROBOT_WORLD_ROBOT_DRIVER_HPP  // 那就定义它，并开始写代码

#include <memory>
#include <vector>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
// #include <sensor_msgs/msg/image.hpp> // Python直接订阅图像信息，避免C++转手延迟
#include <std_msgs/msg/float32_multi_array.hpp> // 用于发送状态数组给 Python
// 引入 TF2 库用于四元数转欧拉角
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class RobotDriver : public rclcpp::Node
{
public:
  RobotDriver();

private:
  // --- 参数配置结构体 ---
  struct RobotConfig
  {
    double dist_from_collision;    // 防撞距离
    double roll_tilt_thresh;       // 翻滚角阈值
    double pitch_tilt_thresh;      // 俯仰角阈值
    double dist_to_goal_thresh;    // 目标容差
    double speed_stop_thresh;      // 停止速度阈值
    int lidar_sector_num;          // 雷达分区数
    double loop_freq;              // 循环频率
  } config_; // 直接实例化一个成员变量

  // --- 回调函数 ---

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

  void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);

  void rl_cmd_callback(const geometry_msgs::msg::Twist::SharedPtr msg);

  void timer_callback();

  void check_safety_and_control();

  void stop_robot();
  // --- 核心逻辑：打包状态给 RL ---
  void publish_state();

  // --- 成员变量 ---
  // Publishers & Subscribers
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr state_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr rl_cmd_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // State Data
  double current_x_ = 0, current_y_ = 0, current_theta_ = 0;
  double current_v_ = 0, current_w_ = 0;
  double current_roll_ = 0, current_pitch_ = 0, current_yaw_imu_ = 0, current_acc_z_ = 0;
  double goal_x_ = 0, goal_y_ = 0;
  bool goal_set_;

  std::vector<float> processed_scan_;
  float last_min_dist_;
};
