import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='ezImage',
     version='0.1',
     scripts=['ezImage'] ,
     author="Milad Pourrahmani (Miladiouss)",
     author_email="miladiouss@gmail.com",
     description="Load and display images and access its content data with one-liners. A PIL wrapper ideal for machine learning and image processing.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/miladiouss/ezImage",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
