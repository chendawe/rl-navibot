from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']), # ★ 自动找到 perception/ 和 perception.slam/
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'scikit-image', 'scipy', 'numpy', 'matplotlib'],
    zip_safe=True,
    maintainer='chendawww',
    maintainer_email='chendawww@todo.todo',
    description='SLAM topology extraction (DRG) package',
    license='MIT',
    # 即使你现在不写 ROS2 节点，这里也要写个空列表占位，否则 colcon 可能报错
    entry_points={
        'console_scripts': [],
    },
)
