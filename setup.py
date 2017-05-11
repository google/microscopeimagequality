import setuptools

setuptools.setup(
    install_requires=[

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
