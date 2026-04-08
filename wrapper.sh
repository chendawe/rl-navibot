mkdir -p ~/.local/share/jupyter/kernels/ros2

cat > ~/.local/share/jupyter/kernels/ros2/start.sh << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
exec /home/chendawww/Software/anaconda3/envs/ros2/bin/python -m ipykernel_launcher "$@"
EOF
chmod +x ~/.local/share/jupyter/kernels/ros2/start.sh

cat > ~/.local/share/jupyter/kernels/ros2/kernel.json << 'EOF'
{
  "argv": ["/home/chendawww/.local/share/jupyter/kernels/ros2/start.sh", "-f", "{connection_file}"],
  "display_name": "ROS2 Humble",
  "language": "python",
  "metadata": {"debugger": true},
  "kernel_protocol_version": "5.5"
}
EOF
