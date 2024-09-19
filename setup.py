from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="feature_selection_ensemble", 
    version="0.1.0",  # 版本号
    author="FaithHuang", 
    author_email="huangfaith317@gmail.com",
    description="A Python package for feature selection ensemble algorithms in tumor gene subset selection",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    url="https://github.com/faithwangsahye/ML-Integrated-Models-for-Tumor-Feature-Selection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: MIT License",  #项目许可证还没改
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[ # 依赖包，看看还有没有需要添加的
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "matplotlib",
        "tqdm",
        "skfeature",
        "arfs",
    ],
    #允许定义可执行脚本
    entry_points={  
        "console_scripts": [
            "my_command=feature_selection_ensemble.some_module:main_function"
        ],
    },
)
