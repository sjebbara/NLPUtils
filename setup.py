from setuptools import setup
from setuptools import find_packages


setup(name='nlputils',
      version='0.1.0',
      description='Useful tools and functions for NLP research',
      author='Soufian Jebbara',
      author_email='s.jebbara@gmail.com',
      url='https://github.com/sjebbara/nlputils',
      download_url='https://github.com/sjebbara/nlputils/tarball/0.3.1',
      license='MIT',
      install_requires=['keras', 'gensim'],
      packages=find_packages())
