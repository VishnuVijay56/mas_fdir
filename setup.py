from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'agent_measurements'

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
        	'interagent_measurements = agent_measurements.interagent_measurements:main',
        	'fixed_frames_tf2_broadcaster = agent_measurements.fixed_frames_tf2_broadcaster:main',
        	'dynamic_frames_tf2_broadcaster = agent_measurements.dynamic_frames_tf2_broadcaster:main',
        	
        ],
    },
)
