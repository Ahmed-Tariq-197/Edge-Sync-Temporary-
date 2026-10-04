import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'edgesync_acquisition'

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
    description='Target Frustum Lock and Liveness Publisher',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'target_acquisition_node = edgesync_acquisition.nodes.target_acquisition_node:main',
            'liveness_publisher = edgesync_acquisition.nodes.liveness_publisher:main',
        ],
    },
)
