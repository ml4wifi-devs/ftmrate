from setuptools import setup, find_packages

setup(
    name='ml4wifi',
    version='1.0.0',
    packages=find_packages(include=[
        'ml4wifi',
        'ml4wifi.*'
    ]),
    install_requires=[
        # pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html,
        'matplotlib~=3.6.3',
        # 'ns3-ai==1.0.2',
        'numpy~=1.23.5',
        'pandas~=1.5.3',
        'scipy~=1.10.0',
        'sympy~=1.11.1',
        'tensorflow-probability~=0.19.0',
    ],
)
