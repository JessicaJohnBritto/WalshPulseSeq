# from setuptools import setup, find_packages

# def parse_requirements(filename):
#     with open(filename, 'r') as file:
#         return file.read().splitlines()

# setup(
# 	name = 'WalshPulseSeq',
# 	version = '0.1',
# 	packages = find_packages(),
# 	#install_requires=parse_requirements('requirements_for_pip.txt'),
#     include_package_data=True,
#     description='A package for Implementing Walsh Pulse Sequenece based on XY Hamiltonian',
#     author='JessicaJohnBritto',
#     classifiers=[
#         'Programming Language :: Python :: 3',
#         'License :: OSI Approved :: MIT License',
#         'Operating System :: OS Independent',
#     ],
#     python_requires='>=3.6',
# )

from setuptools import setup, find_packages
setup(
    name='WPSProtocol',
    version='0.1.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy==1.24.3',
        'matplotlib==3.2.2',
        'scipy==1.10.1',
    ],
    author='Jessica John Britto',
    readme = 'README.md',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='==3.8.19',
    description='A package for Implementing Walsh Pulse Sequenece constructed out of XY Hamiltonian based on Votto, M., Zeiher, J., & Vermersch, B. (2023). Robust universal quantum processors in spin systems via Walsh pulse sequences. arXiv preprint arXiv:2311.10600.',
)
