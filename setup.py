from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
	name = 'WalshPulseSeq',
	version = '0.1',
	packages = find_packages(),
	#install_requires=parse_requirements('requirements_for_pip.txt'),
    include_package_data=True,
    description='A package for Implementing Walsh Pulse Sequenece based on XY Hamiltonian',
    author='JessicaJohnBritto',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
