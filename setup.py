from setuptools import setup, find_packages
import sys

sys.path.append('./chemvaewrapper')

setup(name='ChemVAEWrapper',
        version='2021.5.19',
        description='ChemVAEWrapper',
        long_description="README",
        author='Kan Hatakeyama',
        license=license,
        packages = find_packages(),
    )