'''
Entry points enables you to call the program from
command prompt simply as exhibit, from any directory.

If installing as a developer using "pip install -e ."
don't delete exhibit.egg-info folder.
'''


from setuptools import setup, find_packages

setup(name="exhibit",
      version="0.1",
      description="Command line tool for generating demonstrator data",
      author="German Priks",
      python_requires=">=3.8",
      install_requires=[
        "pandas",
        "numpy",
        "PyYAML"
      ],
      packages=find_packages(),
      entry_points={
        "console_scripts": [
            "exhibit = exhibit.command.bootstrap:main"
        ]}
     )
