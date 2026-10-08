from setuptools import setup
from glob import glob
import os

package_name = 'openflex_isaac_bringup'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name] if os.path.exists(package_name) else [],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenFleX Team',
    maintainer_email='openflex@example.com',
    description='OpenFleX Isaac Sim Integration',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
