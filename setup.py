from setuptools import setup, find_packages

setup(
    name='ml4wifi',
    version='1.0.0',
    packages=find_packages(include=[
        'ml4wifi',
        'ml4wifi.*'
    ]),
    install_requires=[
        'chex~=0.1.5',
        'jax~=0.3.20',
        'jaxlib~=0.3.20',
        # pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html,
        'matplotlib~=3.5.3',
        # 'ns3-ai==1.0.2',
        'optax~=0.1.3',
        'pandas~=1.5.0',
        'scipy~=1.9.1',
        'seaborn~=0.12.0',
        'sympy~=1.11.1',
        'tensorflow-probability~=0.18.0',
        'tqdm~=4.64.1',
    ],
)
