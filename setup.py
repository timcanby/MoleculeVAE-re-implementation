from setuptools import setup

setup(
    name='MoleculeVAE-re-implementation',
    version='1.0.0',
    description='MolecularVAE',
    packages=['MoleculeVAE-reImplementation'],
    url='https://github.com/timcanby/MoleculeVAE-re-implementation',
    license='',
    author='kangyingli',
    author_email='kangyinglee0501@gmail.com',
    install_requires=[
        'setuptools == 49.2.1',
        'deepchem == 2.6.1',
        'h5py == 2.10.0',
        'numpy == 1.22.4',
        'pandas == 1.2.4',
        'rdkit == 2022.3.5',
        'rdkit_pypi == 2022.3.5',
        'scikit_learn == 1.1.2',
        'seaborn == 0.11.1',
        'torch == 1.9.0',
    ],
    keywords='development, practice, structure',
    python_requires='>=3.7, <4',

)
