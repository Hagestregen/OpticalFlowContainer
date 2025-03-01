from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nueflow'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include non-Python data files (e.g., model checkpoints)
        (os.path.join('share', package_name, 'NeuFlow_v2_master'), [
            'nueflow/NeuFlow_v2_master/neuflow_mixed.pth',
            'nueflow/NeuFlow_v2_master/neuflow_sintel.pth',
            'nueflow/NeuFlow_v2_master/neuflow_things.pth',
        ]),
    ],
    package_data={
        # Include all Python files in NeuFlow_v2_master/NeuFlow/data_utils
        package_name: ['NeuFlow_v2_master/data_utils/*.py'],
    },
    install_requires=[
        'setuptools',
        'torch>=1.9.0',
        'numpy<2.0.0',
        'opencv-python',
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