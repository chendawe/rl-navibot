#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>
#include <gazebo_msgs/srv/set_entity_state.hpp>
#include <geometry_msgs/msg/pose.hpp>

class WorldManager : public rclcpp::Node
{
public:
  explicit WorldManager(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void handle_reset(
    const std::shared_ptr<std_srvs::srv::Empty::Request> request,
    std::shared_ptr<std_srvs::srv::Empty::Response> response);

  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;
  rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr gazebo_client_;
};
