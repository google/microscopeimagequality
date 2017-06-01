import setuptools

setuptools.setup(
    entry_points={
        "console_scripts": [
            "quality=quality.application:command"
        ]
    },
    install_requires=[
        "click",
        "matplotlib",
        "nose",
        "numpy",
        "pillow",
        "pytest",
        "scikit-image",
        "scipy",
        "six",
        "tensorflow"
    ],
    name="quality",
    package_data={
        "quality": [
            "data/model/model.ckpt-*"
        ]
    },
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    version="0.1.0"
)
