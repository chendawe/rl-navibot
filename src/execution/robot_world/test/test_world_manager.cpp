// test/test_world_manager.cpp

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>
#include <gazebo_msgs/srv/set_entity_state.hpp>

// 1. 包含头文件
#include "robot_world/world_manager.hpp"

class WorldManagerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);

    // 配置参数：测试是否能正确读取参数
    rclcpp::NodeOptions options;
    options.append_parameter_override("robot_name", "test_bot");
    options.append_parameter_override("reset_x", 5.0);
    options.append_parameter_override("reset_y", 5.0);

    // 实例化被测节点
    manager_node_ = std::make_shared<WorldManager>(options);

    // 实例化测试节点
    test_node_ = std::make_shared<rclcpp::Node>("test_helper");

    // --- 关键：模拟 Gazebo 服务端 ---
    // 因为没有真 Gazebo，我们建一个假的服务来接收请求
    mock_gazebo_server_ = test_node_->create_service<gazebo_msgs::srv::SetEntityState>(
      "/gazebo/set_entity_state",
      [this](const std::shared_ptr<gazebo_msgs::srv::SetEntityState::Request> req,
      std::shared_ptr<gazebo_msgs::srv::SetEntityState::Response> res) {
        received_request_ = req; // 保存收到的请求
        res->success = true;     // 假装成功了
        mock_server_called_ = true;
      });

    // 客户端：用来调用重置服务
    reset_client_ = test_node_->create_client<std_srvs::srv::Empty>("/reset_robot");
  }

  void TearDown() override
  {
    rclcpp::shutdown();
  }

  void spin_some()
  {
    rclcpp::spin_some(manager_node_);
    rclcpp::spin_some(test_node_);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // --- 成员变量 ---
  std::shared_ptr<WorldManager> manager_node_;
  std::shared_ptr<rclcpp::Node> test_node_;

  rclcpp::Service<gazebo_msgs::srv::SetEntityState>::SharedPtr mock_gazebo_server_;
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr reset_client_;

  std::shared_ptr<gazebo_msgs::srv::SetEntityState::Request> received_request_;
  bool mock_server_called_ = false;
};

// === 测试用例 1: 测试重置流程 ===
TEST_F(WorldManagerTest, TestResetServiceCall)
{
  // 1. 发送重置请求
  auto req = std::make_shared<std_srvs::srv::Empty::Request>();
  reset_client_->async_send_request(req);

  // 2. 循环等待，直到假 Gazebo 服务被调用
  mock_server_called_ = false;
  auto start = std::chrono::steady_clock::now();
  while (!mock_server_called_ &&
    std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
  {
    spin_some();
  }

  // 3. 验证结果
  ASSERT_TRUE(mock_server_called_) << "WorldManager did not call the Gazebo service!";
  ASSERT_TRUE(received_request_ != nullptr);
}

// === 测试用例 2: 测试参数传递 ===
TEST_F(WorldManagerTest, TestParameterPassing)
{
  // 发送请求
  reset_client_->async_send_request(std::make_shared<std_srvs::srv::Empty::Request>());

  // 等待
  mock_server_called_ = false;
  auto start = std::chrono::steady_clock::now();
  while (!mock_server_called_ &&
    std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
  {
    spin_some();
  }

  ASSERT_TRUE(mock_server_called_);

  // 验证请求里的参数是否和我们 SetUp 时设置的一样
  // 我们设置了 robot_name="test_bot", x=5.0, y=5.0
  EXPECT_EQ(received_request_->state.name, "test_bot");
  EXPECT_DOUBLE_EQ(received_request_->state.pose.position.x, 5.0);
  EXPECT_DOUBLE_EQ(received_request_->state.pose.position.y, 5.0);
  EXPECT_DOUBLE_EQ(received_request_->state.pose.orientation.w, 1.0);
}


// #include <gtest/gtest.h>
// #include <rclcpp/rclcpp.hpp>
// #include <std_srvs/srv/empty.hpp>
// #include <gazebo_msgs/srv/set_entity_state.hpp>

// // 假设你的类名是 WorldManager
// // #include "your_package/world_manager.hpp"

// // 为了演示，这里假设 WorldManager 类已经在上下文中可用

// class WorldManagerTest : public ::testing::Test
// {
// protected:
//   void SetUp() override
//   {
//     rclcpp::init(0, nullptr);

//     // 1. 初始化被测节点
//     // 这里我们可以传参数，比如测试 Burger
//     rclcpp::NodeOptions options;
//     options.append_parameter_override("robot_name", "test_robot");
//     options.append_parameter_override("reset_x", 5.0); // 测试重置到 x=5.0

//     manager_node_ = std::make_shared<WorldManager>(options);

//     // 2. 创建辅助测试节点
//     test_node_ = std::make_shared<rclcpp::Node>("test_helper");

//     // 3. 创建一个“假的” Gazebo 服务端
//     // 用来捕获 WorldManager 发出的请求
//     mock_gazebo_server_ = test_node_->create_service<gazebo_msgs::srv::SetEntityState>(
//       "/gazebo/set_entity_state",
//       [this](const std::shared_ptr<gazebo_msgs::srv::SetEntityState::Request> req,
//              std::shared_ptr<gazebo_msgs::srv::SetEntityState::Response> res) {
//         received_request_ = req; // 捕获请求内容
//         res->success = true;     // 假装成功了
//         mock_server_called_ = true;
//       });

//     // 4. 创建客户端来调用 /reset_robot
//     reset_client_ = test_node_->create_client<std_srvs::srv::Empty>("/reset_robot");
//   }

//   void TearDown() override
//   {
//     rclcpp::shutdown();
//   }

//   // 辅助函数：让节点跑一会儿
//   void spin_until_called(int timeout_ms = 2000)
//   {
//     mock_server_called_ = false;
//     auto start = std::chrono::steady_clock::now();

//     while (!mock_server_called_ &&
//            std::chrono::steady_clock::now() - start < std::chrono::milliseconds(timeout_ms))
//     {
//       rclcpp::spin_some(manager_node_);
//       rclcpp::spin_some(test_node_);
//       std::this_thread::sleep_for(std::chrono::milliseconds(10));
//     }
//   }

//   // 成员变量
//   std::shared_ptr<WorldManager> manager_node_;
//   std::shared_ptr<rclcpp::Node> test_node_;

//   rclcpp::Service<gazebo_msgs::srv::SetEntityState>::SharedPtr mock_gazebo_server_;
//   rclcpp::Client<std_srvs::srv::Empty>::SharedPtr reset_client_;

//   // 用于存储捕获到的请求
//   std::shared_ptr<gazebo_msgs::srv::SetEntityState::Request> received_request_;
//   bool mock_server_called_ = false;
// };

// // --- 测试用例 1: 检查是否正确调用了 Gazebo 服务 ---
// TEST_F(WorldManagerTest, TestResetServiceCall)
// {
//   // 1. 发送重置请求
//   auto req = std::make_shared<std_srvs::srv::Empty::Request>();
//   auto future = reset_client_->async_send_request(req);

//   // 2. 等待处理
//   spin_until_called();

//   // 3. 验证：假的 Gazebo 服务端应该被调用了
//   ASSERT_TRUE(mock_server_called_) << "WorldManager did not call the Gazebo service!";
//   ASSERT_TRUE(received_request_ != nullptr);
// }

// // --- 测试用例 2: 检查参数传递是否正确 (模型名、位置) ---
// TEST_F(WorldManagerTest, TestResetParameters)
// {
//   // 1. 发送重置请求
//   auto req = std::make_shared<std_srvs::srv::Empty::Request>();
//   reset_client_->async_send_request(req);

//   // 2. 等待
//   spin_until_called();

//   // 3. 验证捕获的请求数据
//   ASSERT_TRUE(mock_server_called_);

//   // 我们在 SetUp 里设置的参数是 "test_robot" 和 x=5.0
//   EXPECT_EQ(received_request_->state.name, "test_robot");
//   EXPECT_DOUBLE_EQ(received_request_->state.pose.position.x, 5.0);
//   EXPECT_DOUBLE_EQ(received_request_->state.pose.position.y, 0.0);
//   EXPECT_DOUBLE_EQ(received_request_->state.pose.orientation.w, 1.0);
// }
