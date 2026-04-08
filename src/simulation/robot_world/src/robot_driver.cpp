
#include "robot_world/robot_driver.hpp"

// 2. 构造函数实现：注意前面加了 RobotDriver::
RobotDriver::RobotDriver() : Node("robot_driver"), goal_set_(false), last_min_dist_(100.0)
{ 
  // --- 0. 初始化参数配置 ---
  config_.dist_from_collision = 0.25;      
  config_.roll_tilt_thresh = 0.35;          
  config_.pitch_tilt_thresh = 0.35;
  config_.dist_to_goal_thresh = 0.2;       
  config_.speed_stop_thresh = 0.05;   
  config_.lidar_sector_num = 24;         
  config_.loop_freq = 10.0;      
  
  // --- 1. 控制相关 ---
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  rl_cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "/rl_cmd", 10,
    std::bind(&RobotDriver::rl_cmd_callback, this, std::placeholders::_1));

  // --- 2. 状态感知相关 ---
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/odom", 10,
    std::bind(&RobotDriver::odom_callback, this, std::placeholders::_1));

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    "/imu", 10,
    std::bind(&RobotDriver::imu_callback, this, std::placeholders::_1));

  scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", 10,
    std::bind(&RobotDriver::scan_callback, this, std::placeholders::_1));

  goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/goal_pose", 10,
    std::bind(&RobotDriver::goal_callback, this, std::placeholders::_1));

  // --- 3. RL 状态发布 ---
  state_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/robot_state", 10);

  // --- 4. 定时器 ---
  int period_ms = static_cast<int>(1000.0 / config_.loop_freq);
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(period_ms),
    std::bind(&RobotDriver::timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "Robot Driver Initialized.");
}

// 3. 各个成员函数的实现：全部加上 RobotDriver::

void RobotDriver::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  current_x_ = msg->pose.pose.position.x;
  current_y_ = msg->pose.pose.position.y;
  current_v_ = msg->twist.twist.linear.x;
  current_w_ = msg->twist.twist.angular.z;
  
  tf2::Quaternion q(
    msg->pose.pose.orientation.x,
    msg->pose.pose.orientation.y,
    msg->pose.pose.orientation.z,
    msg->pose.pose.orientation.w);
  current_theta_ = tf2::getYaw(q);
}

void RobotDriver::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  current_w_ = msg->angular_velocity.z;
  tf2::Quaternion q(
    msg->orientation.x,
    msg->orientation.y,
    msg->orientation.z,
    msg->orientation.w);
  tf2::Matrix3x3 m(q);
  m.getRPY(current_roll_, current_pitch_, current_yaw_imu_);
  current_acc_z_ = msg->linear_acceleration.z;
}

void RobotDriver::goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  goal_x_ = msg->pose.position.x;
  goal_y_ = msg->pose.position.y;
  goal_set_ = true;
  RCLCPP_INFO(this->get_logger(), "New Goal Received: (%.2f, %.2f)", goal_x_, goal_y_);
}

void RobotDriver::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  int lidar_sector_num = config_.lidar_sector_num;
  processed_scan_.resize(lidar_sector_num);
  int sector_size = msg->ranges.size() / lidar_sector_num;
  if(sector_size == 0) sector_size = 1;
  
  for (int i = 0; i < lidar_sector_num; i++) {
    float min_dist = msg->range_max;
    for (int j = 0; j < sector_size; j++) {
      size_t idx = static_cast<size_t>(i * sector_size + j); // 改成 size_t
      if(idx >= msg->ranges.size()) break; // 这样就不会报警告了
      float r = msg->ranges[idx];
      if (!std::isinf(r) && r > msg->range_min && r < min_dist) {
        min_dist = r;
      }
    }
    processed_scan_[i] = min_dist;
  }
  last_min_dist_ = *std::min_element(processed_scan_.begin(), processed_scan_.end());
}

void RobotDriver::rl_cmd_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  if (last_min_dist_ < config_.dist_from_collision && msg->linear.x > 0) {
      stop_robot();
      return;
  }

  if (std::abs(current_roll_) > config_.roll_tilt_thresh || std::abs(current_pitch_) > config_.pitch_tilt_thresh) {
      RCLCPP_WARN(this->get_logger(), "Tilt Warning!");
      stop_robot();
      return;
  }
  cmd_vel_pub_->publish(*msg);
}

void RobotDriver::timer_callback()
{
  check_safety_and_control();
  publish_state();
}

void RobotDriver::check_safety_and_control()
{
  if (!goal_set_) {
      stop_robot();
      return;
  }

  float dx = goal_x_ - current_x_;
  float dy = goal_y_ - current_y_;
  float dist_to_goal = std::sqrt(dx*dx + dy*dy);

  if (dist_to_goal < config_.dist_to_goal_thresh && std::abs(current_v_) < config_.speed_stop_thresh) {
      RCLCPP_INFO(this->get_logger(), "Goal Reached!");
      goal_set_ = false; 
      stop_robot();
  }
}

void RobotDriver::stop_robot() {
  auto stop_cmd = geometry_msgs::msg::Twist();
  cmd_vel_pub_->publish(stop_cmd);
}

void RobotDriver::publish_state()
{
  auto state_msg = std_msgs::msg::Float32MultiArray();
  
  state_msg.data.push_back(current_x_);
  state_msg.data.push_back(current_y_);
  state_msg.data.push_back(current_theta_);
  state_msg.data.push_back(current_v_);
  state_msg.data.push_back(current_w_);

  if (goal_set_) {
    float dx = goal_x_ - current_x_;
    float dy = goal_y_ - current_y_;
    float dist = std::sqrt(dx*dx + dy*dy);
    float angle = std::atan2(dy, dx) - current_theta_;
    state_msg.data.push_back(dist);
    state_msg.data.push_back(angle);
  } else {
    state_msg.data.push_back(0.0f);
    state_msg.data.push_back(0.0f);
  }

  if (!processed_scan_.empty()) {
    state_msg.data.insert(state_msg.data.end(), processed_scan_.begin(), processed_scan_.end());
  } else {
    for(int i=0; i<config_.lidar_sector_num; i++) state_msg.data.push_back(0.0f);
  }
  state_pub_->publish(state_msg);
}


// class RobotDriver : public rclcpp::Node
// {
// public:
//   RobotDriver() : Node("robot_driver"), goal_set_(false), last_min_dist_(100.0)
//   { 
//     // --- 0. 初始化参数配置 ---
//     // [m] 防撞距离 (建议设为 robot_radius + 0.1m)
//     config_.dist_from_collision = 0.25;      
//     // [rad] 翻车倾斜极限 (~20度)
//     config_.roll_tilt_thresh = 0.35;          // 修正：roll -> roll
//     config_.pitch_tilt_thresh = 0.35;
//     // [m] 到达目标的容差半径
//     config_.dist_to_goal_thresh = 0.2;       
//     // [m/s] 静止判定速度
//     config_.speed_stop_thresh = 0.05;   
//     // [个] 雷达分区数量
//     config_.lidar_sector_num = 24;         
//     // [Hz] 控制频率
//     config_.loop_freq = 10.0;      
    
//     // --- 1. 控制相关 ---
//     cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
//     rl_cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
//       "/rl_cmd", 10,
//       std::bind(&RobotDriver::rl_cmd_callback, this, std::placeholders::_1));

//     // --- 2. 状态感知相关 ---
    
//     // 订阅里程计 (位置、速度)
//     odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
//       "/odom", 10,
//       std::bind(&RobotDriver::odom_callback, this, std::placeholders::_1));

//     // 订阅 IMU (角速度、姿态)
//     imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
//       "/imu", 10,
//       std::bind(&RobotDriver::imu_callback, this, std::placeholders::_1));

//     // 订阅激光雷达
//     scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
//       "/scan", 10,
//       std::bind(&RobotDriver::scan_callback, this, std::placeholders::_1));

//     // 订阅目标点 (来自 Planning 层)
//     goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
//       "/goal_pose", 10,
//       std::bind(&RobotDriver::goal_callback, this, std::placeholders::_1));

//     // --- 3. RL 状态发布 ---
//     // 发布一个数组给 Python，包含：[x, y, theta, v, w, dist_to_goal, angle_to_goal, laser_data...]
//     state_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/robot_state", 10);

//     // --- 4. 定时器：定期发布状态给 Python ---
//     // 比如 10Hz，根据你的 RL 训练频率调整
//     int period_ms = static_cast<int>(1000.0 / config_.loop_freq);
//     timer_ = this->create_wall_timer(
//       std::chrono::milliseconds(period_ms),
//       std::bind(&RobotDriver::timer_callback, this));

//     RCLCPP_INFO(this->get_logger(), "robot Driver Initialized with State being published.");
//   }

// private:
//   // --- 参数配置结构体 ---
//   struct RobotConfig {
//     double dist_from_collision;    // 防撞距离
//     double roll_tilt_thresh;       // 翻滚角阈值
//     double pitch_tilt_thresh;      // 俯仰角阈值
//     double dist_to_goal_thresh;    // 目标容差
//     double speed_stop_thresh;      // 停止速度阈值
//     int lidar_sector_num;          // 雷达分区数
//     double loop_freq;              // 循环频率
//   } config_; // 直接实例化一个成员变量

//   // --- 回调函数 ---

//   void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
//   {
//     current_x_ = msg->pose.pose.position.x;
//     current_y_ = msg->pose.pose.position.y;
//     current_v_ = msg->twist.twist.linear.x;
//     current_w_ = msg->twist.twist.angular.z;
    
//     // 使用 TF2 将四元数转换为欧拉角
//     tf2::Quaternion q(
//       msg->pose.pose.orientation.x,
//       msg->pose.pose.orientation.y,
//       msg->pose.pose.orientation.z,
//       msg->pose.pose.orientation.w);
//     current_theta_ = tf2::getYaw(q);
//     // tf2::Matrix3x3 m(q);
//     // double roll, pitch, yaw;
//     // m.getRPY(roll, pitch, yaw);
//     // current_theta_ = yaw;
//   }

//   void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
//   {
//     // IMU 主要提供更准确的角速度和姿态
//     // 可以用来修正 odom 的 drift，或者直接用 imu 的 angular_velocity
//     current_w_ = msg->angular_velocity.z;

//       // 2. 提取欧拉角 (用于翻车检测)
//     tf2::Quaternion q(
//       msg->orientation.x,
//       msg->orientation.y,
//       msg->orientation.z,
//       msg->orientation.w);
//     tf2::Matrix3x3 m(q);
//     m.getRPY(current_roll_, current_pitch_, current_yaw_imu_);

//     // 3. 提取 Z 轴加速度 (用于颠簸/失重检测)
//     current_acc_z_ = msg->linear_acceleration.z;
//   }

//   void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
//   {
//     goal_x_ = msg->pose.position.x;
//     goal_y_ = msg->pose.position.y;
//     goal_set_ = true;
//     RCLCPP_INFO(this->get_logger(), "New Goal Received: (%.2f, %.2f)", goal_x_, goal_y_);
//   }

//   void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
//   {
//     // 【核心简化】将 360 个点降维为 24 个扇区
//     // 这样 Python 端接收到的数据量小，训练快
//     int lidar_sector_num = config_.lidar_sector_num;
//     processed_scan_.resize(lidar_sector_num);
//     int sector_size = msg->ranges.size() / lidar_sector_num;
    
//     for (int i = 0; i < lidar_sector_num; i++) {
//       float min_dist = msg->range_max;
//       for (int j = 0; j < sector_size; j++) {
//         float r = msg->ranges[i * sector_size + j];
//         if (!std::isinf(r) && r > msg->range_min && r < min_dist) {
//           min_dist = r;
//         }
//       }
//       processed_scan_[i] = min_dist;
//     }
    
//     // 同时更新最近障碍物距离（用于安全保护）
//     last_min_dist_ = *std::min_element(processed_scan_.begin(), processed_scan_.end());
//   }


//   void rl_cmd_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
//   {
//     // // 1. 极度危险距离 (< 0.15m)：只允许倒车，禁止任何前进或原地转圈（可能会蹭到）
//     // if (last_min_dist_ < 0.15) {
//     //     if (msg->linear.x > 0) {
//     //         stop_robot(); // 禁止前进
//     //         return;
//     //     }
//     //     // 允许倒车或转弯，通过
//     //     cmd_vel_pub_->publish(*msg);
//     //     return;
//     // }

//     // 安全保护 1：撞墙检测
//     if (last_min_dist_ < config_.dist_from_collision && msg->linear.x > 0) {
//         stop_robot();
//         return;
//     }

//     // 安全保护 2：翻车检测
//     // 如果倾斜超过 20 度 (0.35 rad)，禁止动作
//     if (std::abs(current_roll_) > config_.roll_tilt_thresh || std::abs(current_pitch_) > config_.pitch_tilt_thresh) {
//         RCLCPP_WARN(this->get_logger(), "Tilt Warning! Roll: %.2f, Pitch: %.2f", current_roll_, current_pitch_);
//         stop_robot();
//         return;
//     }

//     cmd_vel_pub_->publish(*msg);
//   }

//   void timer_callback()
//   {
//     // 在这里按顺序调用你想要执行的逻辑
//     check_safety_and_control(); // 步骤A：安全检查与控制
//     publish_state();            // 步骤B：发布状态给 Python
//   }

//   void check_safety_and_control()
//   {
//     // 1. 逻辑：如果没有目标，强制停止
//     if (!goal_set_) {
//         stop_robot();
//         return; // 没目标直接返回
//     }

//     // 2. 新增：检查是否到达目标
//     float dx = goal_x_ - current_x_;
//     float dy = goal_y_ - current_y_;
//     float dist_to_goal = std::sqrt(dx*dx + dy*dy);

//     // 阈值设定：距离 < 0.2米，且线速度很小
//     if (dist_to_goal < config_.dist_to_goal_thresh && std::abs(current_v_) < config_.speed_stop_thresh) {
//         RCLCPP_INFO(this->get_logger(), "Goal Reached! Stopping.");
//         goal_set_ = false; // 重置标志位！
//         stop_robot();
//     }
//   }

//   void stop_robot() {
//     auto stop_cmd = geometry_msgs::msg::Twist();
//     cmd_vel_pub_->publish(stop_cmd);
//   }

//   // --- 核心逻辑：打包状态给 RL ---
//   void publish_state()
//   {
//     auto state_msg = std_msgs::msg::Float32MultiArray();
    
//     // 1. 基本信息
//     state_msg.data.push_back(current_x_);
//     state_msg.data.push_back(current_y_);
//     state_msg.data.push_back(current_theta_);
//     state_msg.data.push_back(current_v_);
//     state_msg.data.push_back(current_w_);

//     // 2. 目标相关信息
//     if (goal_set_) {
//       float dx = goal_x_ - current_x_;
//       float dy = goal_y_ - current_y_;
//       float dist = std::sqrt(dx*dx + dy*dy);
//       float angle = std::atan2(dy, dx) - current_theta_; // 简化计算
      
//       state_msg.data.push_back(dist);       // 距离目标距离
//       state_msg.data.push_back(angle);      // 与目标的角度偏差
//     } else {
//       state_msg.data.push_back(0.0f);
//       state_msg.data.push_back(0.0f);
//     }

//     // 3. 激光雷达数据 (24个点)
//     if (!processed_scan_.empty()) {
//       state_msg.data.insert(state_msg.data.end(), processed_scan_.begin(), processed_scan_.end());
//     } else {
//       // 如果没数据，填充 0
//       for(int i=0; i<24; i++) state_msg.data.push_back(0.0f);
//     }

//     state_pub_->publish(state_msg);
//   }

//   // --- 成员变量 ---
//   // Publishers & Subscribers
//   rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
//   rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr state_pub_;
//   rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr rl_cmd_sub_;
//   rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
//   rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
//   rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
//   rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
//   rclcpp::TimerBase::SharedPtr timer_;

//   // State Data
//   double current_x_ = 0, current_y_ = 0, current_theta_ = 0;
//   double current_v_ = 0, current_w_ = 0;
//   double current_roll_ = 0, current_pitch_ = 0, current_yaw_imu_ = 0, current_acc_z_ = 0;
//   double goal_x_ = 0, goal_y_ = 0;
//   bool goal_set_;
  
//   std::vector<float> processed_scan_;
//   float last_min_dist_;
// };

// int main(int argc, char ** argv)
// {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<RobotDriver>());
//   rclcpp::shutdown();
//   return 0;
// }
