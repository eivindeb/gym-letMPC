from setuptools import setup

setup(
    name='gym-let-mpc',
    version='0.1.0',
    url="https://github.com/eivindeb/gym-letMPC",
    author="Eivind Bøhn",
    author_email="eivind.bohn@gmail.com",
    description="OpenAI Gym Environment for learning event triggered MPC",
    packages=['gym_let_mpc'],
    package_data={"gym_let_mpc": ["configs/cart_pendulum.json"]},
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    install_requires=[
        "do-mpc>=4.1.0",
        "gym>=0.17.3",
        "matplotlib>=3.3.3",
        "numpy>=1.19.3",
        "scipy>=1.7.2"
    ]
)