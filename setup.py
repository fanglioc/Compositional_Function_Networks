from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='cfn',
    version='0.1.0',
    description='Compositional Function Networks (CFN)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Fang Li',
    author_email='fang.li@oc.edu',
    url='https://github.com/fanglioc/Compositional_Function_Networks',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
