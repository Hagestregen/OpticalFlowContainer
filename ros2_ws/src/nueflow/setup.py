from setuptools import find_packages, setup
import os

package_name = 'nueflow'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include the neuflow_mixed.pth file in the installed package
        (os.path.join('share', package_name, 'NeuFlow_v2_master'), ['nueflow/NeuFlow_v2_master/neuflow_mixed.pth']),
    ],
    install_requires=[
        'setuptools',
        'torch>=1.9.0',  # Specify a minimum version compatible with your needs
        'numpy<2.0.0',   # Ensure compatibility with PyTorch as per earlier fix
        'opencv-python',  # For cv2
    ],
    zip_safe=True,
    maintainer='docker',
    maintainer_email='sivert.hb@outlook.com',
    description='ROS2 package for NeuFlow optical flow computation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'neuflow_node = nueflow.neuflow_node:main',
        ],
    },
)