import os
from glob import glob
from setuptools import setup

package_name = 'edgesync_bringup'


# FIX B16 (2026-10-16): models/ was never installed, so every reid_*_path in
# config/algorithm_thresholds.yaml pointed at a file absent from the install space.
# One data_files entry per sub-directory is required on purpose: a single flat
# 'models/' entry would collapse models/transreid/transformer_120.pth to
# models/transformer_120.pth and break the path written in the YAML.
# data/ (68,036 files) and results/ are deliberately NOT installed.
def model_data_files():
    entries = {}
    for path in glob(os.path.join('models', '**', '*'), recursive=True):
        if not os.path.isfile(path):
            continue
        rel = os.path.relpath(os.path.dirname(path), 'models')
        parts = ['share', package_name, 'models'] + ([] if rel == '.' else rel.split(os.sep))
        entries.setdefault(os.path.join(*parts), []).append(path)
    return sorted(entries.items())


MODEL_DATA_FILES = model_data_files()

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ] + MODEL_DATA_FILES,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EdgeSync Team',
    maintainer_email='edgesync-dev@robotics.org',
    description='Launch and configuration package for EdgeSync dual JetAuto UGVs',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
