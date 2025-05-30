from setuptools import find_packages, setup
from glob import glob

package_name = 'pwc_net'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'pwc_node = pwc_net.pwc_node:main',
            'pwc_sub_node = pwc_net.pwc_sub_node:main',
            'sub_n_pub_pwc_junction_node = pwc_net.sub_n_pub_pwc_junction_node:main',
        ],
    },
)
