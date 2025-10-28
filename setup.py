from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dcrnn-pytorch",
    version="0.1.0",
    author="Yaguang Li",
    description="PyTorch implementation of Diffusion Convolutional Recurrent Neural Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/witgaw/DCRNN_PyTorch",
    packages=['dcrnn_pytorch', 'dcrnn_pytorch.lib', 'dcrnn_pytorch.model',
              'dcrnn_pytorch.model.pytorch', 'dcrnn_pytorch.model.tf',
              'dcrnn_pytorch.scripts'],
    package_dir={
        'dcrnn_pytorch': '.',
        'dcrnn_pytorch.lib': 'lib',
        'dcrnn_pytorch.model': 'model',
        'dcrnn_pytorch.scripts': 'scripts',
    },
    py_modules=[],
    install_requires=requirements,
    python_requires=">=3.6",
)
