from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mas_fdir'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vishnu',
    maintainer_email='vvijay@purdue.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'interagent_measurements = mas_fdir.interagent_measurements:main',
        	'fixed_frames_tf2_broadcaster = mas_fdir.fixed_frames_tf2_broadcaster:main',
        	'dynamic_frames_tf2_broadcaster = mas_fdir.dynamic_frames_tf2_broadcaster:main',
        	'fdir_node_centralized = mas_fdir.fdir_node_centralized:main',
        	
        ],
    },
)
