import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'edgesync_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['tests*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='EdgeSync Team',
    maintainer_email='edgesync-dev@robotics.org',
    description='2D-to-3D Visual Perception and Depth Deprojection for EdgeSync',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = edgesync_perception.nodes.perception_node:main',
        ],
    },
)
