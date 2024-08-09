import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="libauc",
  version="1.4.0",
  author="Zhuoning Yuan, Tianbao Yang",
  description="LibAUC: A Deep Learning Library for X-Risk Optimization",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/Optimization-AI/LibAUC",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
  python_requires=">=3.8",
  install_requires = [
                      'torch',
                      'torchvision',
                      'numpy',
                      'tqdm',
                      'scipy',
                      'pandas',
                      'Pillow',
                      'scikit-learn',
                      'opencv-python',
                      'torch_geometric',
                      'ogb',
                      'webdataset']
)
