import setuptools

setuptools.setup(
    python_requires='<3.8',
    entry_points={
        "console_scripts": [
            "microscopeimagequality=microscopeimagequality.application:command"
        ]
    },
    install_requires=[
        "click",
        "matplotlib",
        "nose",
        "numpy<1.19.0,>=1.16.0",
        "Pillow",
        "scikit-image",
        "scipy",
        "six",
        "tensorflow==2.3.1",
        "imagecodecs",
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
    version="0.1.0dev5"
)
