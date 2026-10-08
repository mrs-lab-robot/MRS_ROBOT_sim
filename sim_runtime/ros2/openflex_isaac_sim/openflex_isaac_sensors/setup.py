from setuptools import setup
from glob import glob
import os

package_name = 'openflex_isaac_sensors'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config', 'calibration'), glob('config/calibration/*.yaml')),
        (os.path.join('lib', package_name), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenFleX Team',
    maintainer_email='openflex@example.com',
    description='Sensor integration for OpenFleX in Isaac Sim',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = openflex_isaac_sensors.camera_publisher:main',
            'lidar_publisher = openflex_isaac_sensors.lidar_publisher:main',
            'imu_publisher = openflex_isaac_sensors.imu_publisher:main',
        ],
    },
)
