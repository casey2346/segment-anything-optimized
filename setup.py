from setuptools import setup, find_packages

setup(
    name="segment-anything-optimized",
    use_scm_version=True,  # Enables automatic versioning from Git tags
    setup_requires=["setuptools_scm"],

    author="Hui Li",
    author_email="hui.li.research@example.com",
    description="Reproduction and optimization of Meta AI's Segment Anything Model (SAM) with fine-tuning, deployment, and Hugging Face integration.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hui-li-lab/segment-anything-optimized",
    project_urls={
        "Documentation": "https://github.com/hui-li-lab/segment-anything-optimized",
        "Hugging Face Demo": "https://huggingface.co/spaces/hui-li-lab/sam-demo",
        "Bug Tracker": "https://github.com/hui-li-lab/segment-anything-optimized/issues",
    },

    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    include_package_data=True,

    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.23",
        "scipy>=1.10",
        "Pillow>=9.5",
        "matplotlib>=3.7",
        "opencv-python-headless>=4.7.0",
        "pytorch-lightning>=2.2.1",
        "lightning-utilities>=0.10",
        "torchmetrics>=1.3.2",
        "wandb>=0.16.3",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "python-multipart>=0.0.7",
        "pydantic>=2.7",
        "jupyter>=1.0",
        "nbconvert>=7.10",
        "huggingface_hub>=0.22",
        "onnx>=1.15",
        "onnxruntime>=1.17",
        "tqdm>=4.66",
        "requests>=2.31",
        "pyyaml>=6.0",
        "rich>=13.7",
        "GitPython>=3.1",
        "jsonschema>=4.21"
    ],

    extras_require={
        "demo": ["gradio>=4.25"],
        "notifications": ["slack_sdk>=3.27"],
        "analysis": ["scikit-learn>=1.4"],
        "dev": [
            "pytest>=8.1",
            "autopep8>=2.1",
            "pre-commit>=3.7",
            "packaging>=24.0",
            "setuptools>=70.0",
            "wheel>=0.43"
        ]
    },

    entry_points={
        "console_scripts": [
            "sam-opt=cli.main:main"
        ]
    },

    data_files=[
        ("config", ["configs/config.yaml"]),
        ("models", ["models/sam_vit_b.pth"]),
    ],

    tests_require=["pytest"],

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Build Tools",
    ],

    keywords=[
        "segment-anything", "SAM", "computer vision", "segmentation", "PyTorch", "fine-tuning",
        "huggingface", "wandb", "open source", "mask generation", "object detection"
    ],
)
