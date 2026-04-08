#include "robot_world/world_manager.hpp" // 包含自己的头文件

// 构造函数实现
WorldManager::WorldManager(const rclcpp::NodeOptions & options)
: Node("world_manager", options)
{
  // 1. 声明参数
  this->declare_parameter("robot_name", "turtlebot3_waffle");
  this->declare_parameter("reset_x", 0.0);
  this->declare_parameter("reset_y", 0.0);

  // 2. 创建服务端
  reset_srv_ = this->create_service<std_srvs::srv::Empty>(
    "/reset_robot",
    std::bind(&WorldManager::handle_reset, this, std::placeholders::_1, std::placeholders::_2));

  // 3. 创建客户端
  gazebo_client_ =
    this->create_client<gazebo_msgs::srv::SetEntityState>("/gazebo/set_entity_state");

  RCLCPP_INFO(this->get_logger(), "World Manager Initialized.");
}

// handle_reset 函数实现
void WorldManager::handle_reset(
  const std::shared_ptr<std_srvs::srv::Empty::Request> request,
  std::shared_ptr<std_srvs::srv::Empty::Response> response)
{
  (void)request;
  (void)response;

  // 检查 Gazebo 服务是否准备好
  if (!gazebo_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_ERROR(this->get_logger(), "Gazebo service not available!");
    return;
  }

  // 构造请求
  auto req = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();

  // 读取参数
  req->state.name = this->get_parameter("robot_name").as_string();
  req->state.pose.position.x = this->get_parameter("reset_x").as_double();
  req->state.pose.position.y = this->get_parameter("reset_y").as_double();
  req->state.pose.position.z = 0.0;
  req->state.pose.orientation.w = 1.0;

  // // 发送请求并同步等待
  // auto future_result = gazebo_client_->async_send_request(req);

  // // 阻塞等待结果
  // if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future_result, std::chrono::seconds(2))
  //     == rclcpp::FutureReturnCode::SUCCESS)
  // {
  //   RCLCPP_INFO(this->get_logger(), "Robot reset successfully.");
  // } else {
  //   RCLCPP_ERROR(this->get_logger(), "Failed to reset robot.");
  // }
  // --- 修改：使用异步回调，避免嵌套 Spin 导致崩溃 ---
  auto callback = [this](rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedFuture future) {
      auto result = future.get();
      if (result) {
        RCLCPP_INFO(this->get_logger(), "Robot reset successfully.");
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to reset robot.");
      }
    };
  gazebo_client_->async_send_request(req, callback);
}


// #include <memory>
// #include <rclcpp/rclcpp.hpp>
// #include <std_srvs/srv/empty.hpp>
// #include <gazebo_msgs/srv/set_entity_state.hpp>
// #include <geometry_msgs/msg/pose.hpp>

// class WorldManager : public rclcpp::Node
// {
// public:
//   WorldManager() : Node("world_manager")
//   {
//     // 1. 声明参数，方便 Launch 文件动态配置
//     this->declare_parameter("robot_name", "turtlebot3_waffle");
//     this->declare_parameter("reset_x", 0.0);
//     this->declare_parameter("reset_y", 0.0);

//     // 2. 创建服务端：供 Python 调用重置环境
//     reset_srv_ = this->create_service<std_srvs::srv::Empty>(
//       "/reset_robot",
//       std::bind(&WorldManager::handle_reset, this, std::placeholders::_1, std::placeholders::_2));

//     // 3. 创建客户端
//     gazebo_client_ = this->create_client<gazebo_msgs::srv::SetEntityState>("/gazebo/set_entity_state");

//     RCLCPP_INFO(this->get_logger(), "World Manager Initialized.");
//   }

// private:
//   void handle_reset(
//     const std::shared_ptr<std_srvs::srv::Empty::Request> request,
//     std::shared_ptr<std_srvs::srv::Empty::Response> response)
//   {
//     (void)request;
//     (void)response;

//     // 检查 Gazebo 服务是否准备好
//     if (!gazebo_client_->wait_for_service(std::chrono::seconds(1))) {
//       RCLCPP_ERROR(this->get_logger(), "Gazebo service not available!");
//       return; // 直接返回，Python 端会收到响应（但没报错，建议 Python 端也检查下）
//     }

//     // 构造请求
//     auto req = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();

//     // 从参数服务器读取模型名称
//     req->state.name = this->get_parameter("robot_name").as_string();

//     // 从参数读取位置，方便以后做随机重置
//     req->state.pose.position.x = this->get_parameter("reset_x").as_double();
//     req->state.pose.position.y = this->get_parameter("reset_y").as_double();
//     req->state.pose.position.z = 0.0;
//     req->state.pose.orientation.w = 1.0;

//     // --- 关键改进：同步等待 ---
//     // 只有确信 Gazebo 执行完了，才回复 Python
//     auto future_result = gazebo_client_->async_send_request(req);

//     // 阻塞等待结果 (超时 2 秒)
//     if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future_result, std::chrono::seconds(2))
//         == rclcpp::FutureReturnCode::SUCCESS)
//     {
//       RCLCPP_INFO(this->get_logger(), "Robot reset successfully.");
//     } else {
//       RCLCPP_ERROR(this->get_logger(), "Failed to reset robot.");
//     }
//   }

//   rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;
//   rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr gazebo_client_;
// };
