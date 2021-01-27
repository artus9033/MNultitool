from setuptools import setup, find_packages

# Load long_description from README.md
with open("README.md") as f:
    long_description = f.read()

setup(
    name='mnultitool',
    version='1.0.1',
    description='A multitool for Numerical Methods',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/artus9033/MNultitool',
    author='artus9033',
    author_email='artus9033@gmail.com',
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='numerical,math,interpolation',
    package_dir={'': 'mnultitool'},
    python_requires='>=3.6, <4',
)
