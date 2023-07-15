from setuptools import setup

setup(
    name="QS",
    version="0.0",
    description="Quadrotor Stablization Project",
    author="Xubo, Joe, Hanyang",
    author_email='hha160@sfu.ca',
    # packages=['QS'],
    packages=[
            'adversarial_generation/odp',
            'adversarial_generation/FasTrack_data',
            'phoenix_drone/phoenix_drone_simulation'
           ],
    license="MIT"
)