import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchlm",
    version="0.1.0",
    author="Yan Jun",
    author_email="qyjdef@163.com",
    description="A PyTorch library for landmarks detection, include data augmentation, training and inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DefTruth/torchaug",
    packages=["torchlm"],
    install_requires=[
        "opencv-python>=4.2.1",
        "numpy>=1.14.4",
        "torch>=1.6.0"
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
