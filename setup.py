from setuptools import setup

setup(name='mc_sspals',
      version='0.2.0',
      description='Monte-Carlo simulation of single-shot positron annihilation lifetime spectra',
      url='',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD 3-clause',
      packages=['mc_sspals'],
      install_requires=[
          'scipy>=0.16.1', 'numpy>=1.9.3', 'pandas>=0.9.1'
      ],
      include_package_data=False,
      zip_safe=False)
