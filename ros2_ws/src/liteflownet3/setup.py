from setuptools import find_packages, setup
from glob import glob

package_name = 'liteflownet3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['liteflownet3/network-sintel.pytorch']),
                # Install launch files
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='docker',
    maintainer_email='sivert.hb@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lfn3_node = liteflownet3.lfn3_node:main',
            'depth_calculation_node = liteflownet3.depth_calculation_node:main',
            'optical_flow_spike_viz = liteflownet3.optical_flow_spike_viz:main',
            'depth_subandpub_node = liteflownet3.depth_subandpub_node:main',
            'lfn3_sub_node = liteflownet3.lfn3_sub_node:main',
            'lfn3_junction_node = liteflownet3.lfn3_junction_node:main',
            'sub_n_pub_lfn3_node = liteflownet3.sub_n_pub_lfn3_node:main',
            'lfn3_adapt_node = liteflownet3.lfn3_adapt_node:main',
        ],
    },
)