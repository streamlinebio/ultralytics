from setuptools import setup, find_packages

package_name = 'ultralytics'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=['ultralytics_node', 'ultralytics_node.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sam',
    maintainer_email='sam@example.com',
    description='Ultralytics inference service node for detector pipeline',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ultralytics_node = ultralytics_node.ultralytics_node:main',
        ],
    },
)
