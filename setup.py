from setuptools import setup, find_packages

setup(
    name="im2quant",
    version="0.1.0",
    description="Image-based property prediction using a YOLO backbone",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "ultralytics>=8.0",
        "Pillow>=9.0",
        "numpy>=1.24",
        "pandas>=1.5",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "optuna>=3.0",
        "tqdm>=4.64",
        "matplotlib>=3.7",
    ],
)
