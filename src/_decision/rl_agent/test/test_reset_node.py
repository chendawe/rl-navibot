#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from gazebo_msgs.srv import SetEntityState
import math, time

def main(args=None):
    rclpy.init(args=args)
    node = Node('test_set_entity_state_node')
    cli = node.create_client(SetEntityState, '/set_entity_state')

    # 1. 等服务
    if not cli.wait_for_service(timeout_sec=5.0):
        node.get_logger().error('/set_entity_state not available')
        node.destroy_node()
        rclpy.shutdown()
        return

    # 2. 构造一个最简单的请求
    req = SetEntityState.Request()
    req.state.name = 'waffle'
    req.state.pose.position.x = 0.0
    req.state.pose.position.y = 0.0
    req.state.pose.position.z = 0.1
    req.state.pose.orientation.x = 0.0
    req.state.pose.orientation.y = 0.0
    req.state.pose.orientation.z = 0.0
    req.state.pose.orientation.w = 1.0
    req.state.twist.linear.x = 0.0
    req.state.twist.angular.z = 0.0
    req.state.reference_frame = 'world'

    node.get_logger().info('Calling /set_entity_state ...')
    future = cli.call_async(req)

    # 3. 官方推荐的异步等待方式：在调用线程里用 executor spin_until_future_complete
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    start = time.time()
    while not future.done():
        executor.spin_once(timeout_sec=0.01)
        if time.time() - start > 5.0:
            node.get_logger().error('Timeout!')
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
            return

    # 4. 检查结果
    if future.exception() is not None:
        node.get_logger().error(f'Service call failed: {future.exception()}')
    else:
        res = future.result()
        # 去掉 status_message，只打印 success
        node.get_logger().info(f'Done: success={res.success}')

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
