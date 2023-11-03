from setuptools import setup, find_packages

setup(
    name='ml4wifi',
    version='2.0.0',
    packages=find_packages(include=[
        'ml4wifi',
        'ml4wifi.*'
    ]),
    install_requires=[
        'chex~=0.1.5',
        'jax~=0.4.2',
        'jaxlib~=0.4.2',
        # pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html,
        'matplotlib~=3.6.3',
        # 'ns3-ai==1.0.2',
        'optax~=0.1.3',
        'pandas~=1.5.3',
        'scipy~=1.10.0',
        'seaborn~=0.12.2',
        'sympy~=1.11.1',
        'tensorflow-probability~=0.19.0',
        'tqdm~=4.64.1',
    ],
)
