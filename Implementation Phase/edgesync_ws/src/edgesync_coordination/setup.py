import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'edgesync_coordination'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['tests*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='EdgeSync Team',
    maintainer_email='edgesync-dev@robotics.org',
    description='P2P Negotiation, Ownership FSM, and Anti-Deadlock Consensus',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'coordination_node = edgesync_coordination.nodes.coordination_node:main',
            'run_fault_injection = edgesync_coordination.scripts.run_fault_injection:main',
        ],
    },
)
