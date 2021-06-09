from setuptools import setup, find_packages, Extension

version = '0.1.0'

# g++ ./base/Base.cpp -fPIC -shared -o Base.so -pthread -O3 -march=native

setup(
    name="openke",
    version=version,
    license='MIT',
    description='Pytorch Wrapper of C++ OpenKE-PyTorch',
    author='thunlp',
    url='https://github.com/thunlp/OpenKE',
    packages=find_packages(),
    ext_modules=[
        Extension(
            'base',
            extra_compile_args=["-fPIC", "-shared", "-pthread", "-O3", "-march=native"],
            extra_link_args=["-o", "OpenKEBase.so"],
            sources=['./openke/base/Base.cpp']
        )
    ],
    data_files=[('lib', ['OpenKEBase.so'])],
    install_requires=[
        'numpy==1.16.4',
        'scipy',
        'torch==1.2.0'
    ],
)
