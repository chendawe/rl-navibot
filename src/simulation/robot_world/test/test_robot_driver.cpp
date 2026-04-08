// test/test_robot_driver.cpp
// test/test_robot_driver.cpp

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

// 1. 直接包含你的头文件
#include "robot_world/robot_driver.hpp"

class RobotDriverTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);

    // 实例化被测节点
    driver_node_ = std::make_shared<RobotDriver>();

    // 实例化测试辅助节点 (模拟 Python 端和 Gazebo)
    test_node_ = std::make_shared<rclcpp::Node>("test_helper");

    // --- 模拟发布者 (输入源) ---
    pub_rl_cmd_ = test_node_->create_publisher<geometry_msgs::msg::Twist>("/rl_cmd", 10);
    pub_odom_ = test_node_->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    pub_scan_ = test_node_->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);
    pub_goal_ = test_node_->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 10);

    // --- 模拟订阅者 (接收输出) ---
    sub_cmd_vel_ = test_node_->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        last_cmd_vel_ = *msg;
        cmd_vel_received_ = true;
      });

    sub_state_ = test_node_->create_subscription<std_msgs::msg::Float32MultiArray>(
      "/robot_state", 10,
      [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        last_state_ = *msg;
        state_received_ = true;
      });
    // 稍微 spin 一下，并等待发现连接
    spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 等待 500ms 让 ROS 发现连接
    spin_some();
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }

  // 辅助函数：让 ROS 跑一会儿，处理消息队列
  void spin_some()
  {
    rclcpp::spin_some(driver_node_);
    rclcpp::spin_some(test_node_);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // --- 成员变量 ---
  std::shared_ptr<RobotDriver> driver_node_;
  std::shared_ptr<rclcpp::Node> test_node_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_rl_cmd_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_scan_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_goal_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_state_;

  geometry_msgs::msg::Twist last_cmd_vel_;
  std_msgs::msg::Float32MultiArray last_state_;
  bool cmd_vel_received_ = false;
  bool state_received_ = false;
};
// === 测试用例 1: 正常指令透传 ===
TEST_F(RobotDriverTest, TestNormalPassThrough)
{
  // 1. 发送安全的雷达数据 (距离 10m)
  sensor_msgs::msg::LaserScan scan;
  scan.ranges.resize(360, 10.0);
  scan.range_max = 20.0;
  scan.range_min = 0.1;
  pub_scan_->publish(scan);

  // 先 spin 一下让 scan 生效
  spin_some();

  // 2. 发送 RL 指令
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = 0.5;
  cmd.angular.z = 0.1;
  pub_rl_cmd_->publish(cmd);

  // 3. 等待输出 (修正逻辑：循环检测，直到收到 x=0.5 的指令，或者超时)
  bool success = false;
  auto start = std::chrono::steady_clock::now();

  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
    spin_some();

    // 如果收到了消息
    if (cmd_vel_received_) {
      // 检查数值是否正确
      if (std::abs(last_cmd_vel_.linear.x - 0.5) < 0.01 &&
        std::abs(last_cmd_vel_.angular.z - 0.1) < 0.01)
      {
        success = true;
        break; // 数值对了，退出循环
      }
      // 数值不对（比如是启动时的 0），重置标志位，继续等待下一条
      cmd_vel_received_ = false;
    }
  }

  ASSERT_TRUE(success) << "Did not receive cmd_vel with x=0.5. Last received x: " <<
    last_cmd_vel_.linear.x;
}


// === 测试用例 2: 防撞逻辑 ===
TEST_F(RobotDriverTest, TestCollisionStop)
{
  // 1. 发送危险的雷达数据 (距离 0.1m)
  sensor_msgs::msg::LaserScan scan;
  scan.ranges.resize(360, 10.0);
  scan.ranges[0] = 0.1; // 正前方极近
  scan.range_max = 20.0;
  scan.range_min = 0.1;
  pub_scan_->publish(scan);
  spin_some(); // 更新 last_min_dist_

  // 2. 尝试前进
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = 0.5;
  pub_rl_cmd_->publish(cmd);

  // 3. 等待输出
  cmd_vel_received_ = false;
  auto start = std::chrono::steady_clock::now();
  while (!cmd_vel_received_ && std::chrono::steady_clock::now() - start < std::chrono::seconds(1)) {
    spin_some();
  }

  ASSERT_TRUE(cmd_vel_received_);
  // 预期被 stop_robot() 拦截，速度归零
  EXPECT_FLOAT_EQ(last_cmd_vel_.linear.x, 0.0);
}

// === 测试用例 3: 状态打包测试 ===
TEST_F(RobotDriverTest, TestStatePublishing)
{
  // 1. 发送里程计
  nav_msgs::msg::Odometry odom;
  odom.pose.pose.position.x = 5.0;
  odom.pose.pose.position.y = 3.0;
  odom.pose.pose.orientation.w = 1.0;
  pub_odom_->publish(odom);

  // 2. 发送目标点
  geometry_msgs::msg::PoseStamped goal;
  goal.pose.position.x = 5.0;
  goal.pose.position.y = 5.0;
  pub_goal_->publish(goal);

  spin_some();

  // 3. 等待定时器触发状态发布
  state_received_ = false;
  auto start = std::chrono::steady_clock::now();
  while (!state_received_ && std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
    spin_some();
  }

  ASSERT_TRUE(state_received_);

  // 检查状态数组: [x, y, theta, v, w, dist, angle, scan...]
  EXPECT_FLOAT_EQ(last_state_.data[0], 5.0); // x
  EXPECT_FLOAT_EQ(last_state_.data[1], 3.0); // y
  EXPECT_NEAR(last_state_.data[5], 2.0, 0.01); // dist_to_goal (y差2米)
}


// #include <gtest/gtest.h>
// #include <rclcpp/rclcpp.hpp>
// #include <geometry_msgs/msg/twist.hpp>
// #include <nav_msgs/msg/odometry.hpp>
// #include <sensor_msgs/msg/laser_scan.hpp>
// #include <geometry_msgs/msg/pose_stamped.hpp>
// #include <std_msgs/msg/float32_multi_array.hpp>

// // 假设你的头文件路径是这样的，如果是在单文件测试，直接 include 源文件或声明类
// // #include "robot_driver_pkg/robot_driver.hpp"

// // 为了方便测试，这里直接包含修正后的类定义，或者假设类已经可用
// // 下面演示测试逻辑：

// class RobotDriverTest : public ::testing::Test
// {
// protected:
//   void SetUp() override
//   {
//     rclcpp::init(0, nullptr);

//     // 1. 实例化你的驱动节点
//     driver_node_ = std::make_shared<RobotDriver>();

//     // 2. 实例化测试辅助节点 (用来模拟 Gazebo 和 RL 端)
//     test_node_ = std::make_shared<rclcpp::Node>("test_helper");

//     // --- 模拟发布者 (输入源) ---
//     pub_rl_cmd_ = test_node_->create_publisher<geometry_msgs::msg::Twist>("/rl_cmd", 10);
//     pub_odom_ = test_node_->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
//     pub_scan_ = test_node_->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);
//     pub_goal_ = test_node_->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 10);

//     // --- 模拟订阅者 (接收输出) ---
//     sub_cmd_vel_ = test_node_->create_subscription<geometry_msgs::msg::Twist>(
//       "/cmd_vel", 10,
//       [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
//         last_cmd_vel_ = *msg;
//         cmd_vel_received_ = true;
//       });

//     sub_state_ = test_node_->create_subscription<std_msgs::msg::Float32MultiArray>(
//       "/robot_state", 10,
//       [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
//         last_state_ = *msg;
//         state_received_ = true;
//       });
//   }

//   void TearDown() override
//   {
//     rclcpp::shutdown();
//   }

//   // 辅助函数：让 ROS 转一圈，处理消息
//   void spin_some()
//   {
//     rclcpp::spin_some(driver_node_);
//     rclcpp::spin_some(test_node_);
//     std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 给点时间传输
//   }

//   // --- 成员变量 ---
//   std::shared_ptr<RobotDriver> driver_node_;
//   std::shared_ptr<rclcpp::Node> test_node_;

//   // 发布者
//   rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_rl_cmd_;
//   rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
//   rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_scan_;
//   rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_goal_;

//   // 订阅者
//   rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
//   rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_state_;

//   // 记录结果
//   geometry_msgs::msg::Twist last_cmd_vel_;
//   std_msgs::msg::Float32MultiArray last_state_;
//   bool cmd_vel_received_ = false;
//   bool state_received_ = false;
// };

// // === 测试用例 1: 正常指令透传 ===
// TEST_F(RobotDriverTest, TestNormalPassThrough)
// {
//   // 1. 发送安全的雷达数据 (距离 10m)
//   sensor_msgs::msg::LaserScan scan;
//   scan.ranges.resize(360, 10.0);
//   scan.range_max = 20.0;
//   scan.range_min = 0.1;
//   pub_scan_->publish(scan);
//   spin_some(); // 让 driver 处理 scan

//   // 2. 发送 RL 指令
//   geometry_msgs::msg::Twist cmd;
//   cmd.linear.x = 0.5;
//   cmd.angular.z = 0.1;
//   pub_rl_cmd_->publish(cmd);

//   // 3. 验证输出
//   cmd_vel_received_ = false;
//   auto start = std::chrono::steady_clock::now();
//   while (!cmd_vel_received_ && std::chrono::steady_clock::now() - start < std::chrono::seconds(1)) {
//     spin_some();
//   }

//   ASSERT_TRUE(cmd_vel_received_);
//   EXPECT_FLOAT_EQ(last_cmd_vel_.linear.x, 0.5);
//   EXPECT_FLOAT_EQ(last_cmd_vel_.angular.z, 0.1);
// }

// // === 测试用例 2: 防撞逻辑 ===
// TEST_F(RobotDriverTest, TestCollisionStop)
// {
//   // 1. 发送危险的雷达数据 (距离 0.1m)
//   sensor_msgs::msg::LaserScan scan;
//   scan.ranges.resize(360, 10.0);
//   scan.ranges[0] = 0.1; // 正前方极近
//   scan.range_max = 20.0;
//   scan.range_min = 0.1;
//   pub_scan_->publish(scan);
//   spin_some(); // 更新 last_min_dist_

//   // 2. 尝试前进
//   geometry_msgs::msg::Twist cmd;
//   cmd.linear.x = 0.5; // 前进指令
//   pub_rl_cmd_->publish(cmd);

//   // 3. 验证输出：应该被拦截并发送 0 速度
//   cmd_vel_received_ = false;
//   auto start = std::chrono::steady_clock::now();
//   while (!cmd_vel_received_ && std::chrono::steady_clock::now() - start < std::chrono::seconds(1)) {
//     spin_some();
//   }

//   ASSERT_TRUE(cmd_vel_received_);
//   // 预期被 stop_robot() 处理了，速度归零
//   EXPECT_FLOAT_EQ(last_cmd_vel_.linear.x, 0.0);
// }

// // === 测试用例 3: 状态打包测试 ===
// TEST_F(RobotDriverTest, TestStatePublishing)
// {
//   // 1. 发送里程计
//   nav_msgs::msg::Odometry odom;
//   odom.pose.pose.position.x = 5.0;
//   odom.pose.pose.position.y = 3.0;
//   odom.pose.pose.orientation.w = 1.0; // 朝向 0
//   pub_odom_->publish(odom);

//   // 2. 发送目标点
//   geometry_msgs::msg::PoseStamped goal;
//   goal.pose.position.x = 5.0;
//   goal.pose.position.y = 5.0; // 距离 y方向差 2米
//   pub_goal_->publish(goal);

//   spin_some(); // 让回调处理数据

//   // 3. 手动触发定时器或者等待 (这里我们等待定时器自动触发)
//   state_received_ = false;
//   auto start = std::chrono::steady_clock::now();
//   while (!state_received_ && std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
//     spin_some();
//   }

//   ASSERT_TRUE(state_received_);
//   // 检查状态数组内容: [x, y, theta, v, w, dist, angle, scan...]
//   EXPECT_FLOAT_EQ(last_state_.data[0], 5.0); // x
//   EXPECT_FLOAT_EQ(last_state_.data[1], 3.0); // y

//   // 检查距离目标 (应该约为 2.0)
//   EXPECT_NEAR(last_state_.data[5], 2.0, 0.01);
// }
