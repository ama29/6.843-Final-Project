from setuptools import setup
 
setup(
    name='gripperEnv',
    version = '0.0.1',
    install_requires=[
        'stable-baselines',
        'tensorflow<1.15.0'
        'autopep8',
        'gym',
        'keras==2.2.4',
        'matplotlib',
        'numpy',
        'opencv-contrib-python',
        'pandas',
        'pybullet==2.6.4',
        'pytest',
        'pydot',
        'pyyaml==5.4.1',
        'seaborn',
        'scikit-learn',
        'tqdm',
        'paramiko',
        "h5py<3.0.0",
        "numba",
        "vit-pytorch"
    ],
)
