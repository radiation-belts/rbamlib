from setuptools import setup, find_packages

setup(
    name='rbamlib',
    version='25.04',
    author='Alexander Drozdov',
    author_email='adrozdov@ucla.edu',
    description='A lightweight, open-source Python library for the analysis and modeling of radiation belts.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/radiation-belts/rbamlib',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
