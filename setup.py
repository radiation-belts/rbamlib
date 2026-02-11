from setuptools import setup, find_packages

setup(
    name='rbamlib',
    version='26.02',
    author='Alexander Drozdov',
    author_email='alexander.y.drozdov@aero.org',
    description='A lightweight, open-source Python library for the analysis and modeling of radiation belts.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/radiation-belts/rbamlib',
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
