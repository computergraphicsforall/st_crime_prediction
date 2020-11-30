from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name='st_crime_prediction',
    version='1.0',
    packages=['docs', 'tools', 'data_io', 'notebooks', 'forecast_time_series'],
    url='https://github.com/computergraphicsforall/st_crime_prediction',
    license='',
    author='Miguel Barrero, Jorge Victorino',
    author_email='mbarrerop@gmail.com',
    description='Spatio-temporal crime prediction',
    python_requires='>=3.8'
)
