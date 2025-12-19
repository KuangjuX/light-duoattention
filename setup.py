from setuptools import setup, find_packages
import os


def read_long_description():
    """Read the long description from README.md"""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def read_version():
    """Read version from light_duo_attn/__init__.py"""
    version = {}
    version_path = os.path.join(os.path.dirname(__file__), "light_duo_attn", "kernels", "__init__.py")
    if os.path.exists(version_path):
        with open(version_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    exec(line, version)
                    break
    return version.get("__version__", "0.1.0")


setup(
    name="light-duoattention",
    version=read_version(),
    author="Chengxiang Qi",
    author_email="kuangjux@outlook.com", 
    description="A lightweight CuTe-based CUDA kernel for DuoAttention, optimized for large language model inference",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/KuangjuX/light-duoattention",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch==2.8.0",
        "cuda-python>=12.8.0",
        "nvidia-cutlass-dsl==4.1.0",
        "flash-attn>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-xdist>=3.0.0",
        ],
    },
    keywords=[
        "deep learning",
        "machine learning",
        "attention mechanism",
        "transformer",
        "cuda",
        "gpu",
        "llm",
        "large language models",
        "duoattention",
        "streaming attention",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/light-duoattention/issues",  # 请替换
        "Source": "https://github.com/yourusername/light-duoattention",  # 请替换
    },
    license="Apache-2.0",
    platforms=["Linux"],
    zip_safe=False,
)

