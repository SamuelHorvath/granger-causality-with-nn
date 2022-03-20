import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GCfTSwNN",
    version="0.0.1",
    author="Samuel Horvath",
    author_email="samohorvath11@gmail.com",
    description="Granger Causality for Time Series Modelled by Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamuelHorvath/granger-causality-for-time-series-with-nn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
