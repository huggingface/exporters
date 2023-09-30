from setuptools import setup, find_packages

setup(
    name="exporters",
    version="0.0.1",
    description="Core ML exporter for Hugging Face Transformers",
    long_description="",
    author="The HuggingFace team",
    author_email="matthijs@huggingface.co",
    url="https://github.com/huggingface/exporters",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=[
        "transformers >= 4.30.0",
        "coremltools >= 7",
    ],
    classifiers=[
    ],
    license="Apache",
)
