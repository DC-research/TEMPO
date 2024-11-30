import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def version():
    try:
        return read('VERSION').strip().lstrip('v')
    except:
        return "0.0.0"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

long_description = read('README.md') if os.path.isfile("README.md") else ""

setup(
    name='timeagi',
    version=version(),
    author='Defu Cao',
    author_email='defucao@usc.edu',
    description='Time Series Foundation Model - TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DC-research/TEMPO',
    packages=find_packages(exclude=['schemas', 'tests']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    keywords=['time-series', 'ml', 'llm'],
    python_requires='>=3.7.2,<4',
    install_requires=requirements,
    project_urls={
        'Bug Reports': 'https://github.com/DC-research/TEMPO/issues',
        'Source': 'https://github.com/DC-research/TEMPO',
    },
)