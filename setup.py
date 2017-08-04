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
    name="microscopeimagequality",
    url='https://github.com/google/microscopeimagequality',
    author='Samuel Yang',
    author_email='samuely@google.com',
    license='Apache 2.0',
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    version="0.1.0dev1"
)
