import setuptools

setuptools.setup(
    entry_points={
        "console_scripts": [
            "microscopeimagequality=microscopeimagequality.application:command"
        ]
    },
    install_requires=[
        "click",
        "matplotlib",
        "nose",
        "numpy",
        "Pillow",
        "scikit-image",
        "scipy",
        "six",
        "tensorflow"
    ],
    test_requires=["pytest"],
    name="microscopeimagequality",
    package_data={
        "microscopeimagequality": [
            "data/"
        ]
    },
    classifiers=[
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering'],
    description="Microscope Image Quality Classification",
    url='https://github.com/google/microscopeimagequality',
    author='Samuel Yang',
    author_email='samuely@google.com',
    license='Apache 2.0',
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    version="0.1.0dev2"
)
