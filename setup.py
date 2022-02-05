import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pylandmarks",
    version="0.1.0",
    author="Yan Jun",
    author_email="",
    description="A PyTorch style data augmentation library for landmarks and object detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DefTruth/pylandmarks",
    packages=["pylandmarks"],
    install_requires=[
        "opencv-python>=4.2.1",
        "numpy>=1.14.4",
        "torch>=1.6.0",
        "onnxruntime>=1.7.0"
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
