import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'edgesync_observability'

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
    description='Telemetry Monitoring, Logging and Evaluation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'monitoring_node = edgesync_observability.nodes.monitoring_node:main',
            'profile_edge_latency = edgesync_observability.scripts.profile_edge_latency:main',
            'run_offline_evaluation = edgesync_observability.scripts.run_offline_evaluation:main',
            'scenario_orchestrator = edgesync_observability.scripts.scenario_orchestrator:main',
        ],
    },
)
