from setuptools import setup, find_packages

setup(
    name="algosrl",  
    version="0.1.1",  
    author="LlewynS",  
    author_email="algos251@gmail.com",
    description="Algorithm OS RL: An RL module for AlgOS",
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",
    url="https://github.com/LlewynS/algosrl",  
    packages=find_packages(include=["algosrl", "algosrl.*"]),  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10", 
    install_requires=[
        "algos_core==0.1.0",
        "gym-dcmotor==0.2",
        "gymnasium==0.29.1",
        "numpy==1.23.5",
        "optuna==4.2.1",
        "pytest==7.2.2",
        "setuptools==65.5.0",
        "stable_baselines3==2.4.0",
        "torch==2.0.1",
        "gymnasium_robotics==1.2.3",
        "highway_env==1.8.2"
    ]
)