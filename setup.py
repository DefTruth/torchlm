import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchlm",
    version="0.1.0",
    author="Yan Jun",
    author_email="qyjdef@163.com",
    description="A PyTorch landmarks library with 100+ data augmentation, training and inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DefTruth/torchlm",
    packages=["torchlm"],
    install_requires=[
        "opencv-python-headless>=4.5.2",
        "numpy>=1.14.4",
        "torch>=1.6.0",
        "torchvision>=0.9.0",
        "albumentations>=1.1.0"
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ),
)
