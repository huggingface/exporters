from setuptools import setup, find_packages

setup(
    name="exporters",
    version="0.0.1",
    description="Core ML and TF Lite exporters for Hugging Face Transformers",
    long_description="",
    author="The HuggingFace team",
    author_email="matthijs@huggingface.co",
    url="https://github.com/huggingface/exporters",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.6.0",
    install_requires=[
        "transformers >= 4.26.1",
        "coremltools >= 5.0",
    ],
    classifiers=[
    ],
    license="Apache",
)
