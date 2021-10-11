import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


modules = [
    'code_gen',
    'coupling',
    'graph',
    'ir',
    'mapping',
    'runtime',
    'sim',
    'transformations',
]

setuptools.setup(name='pairs',
    description="A code generator for particle simulations",
    version="0.0.1",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Rafael Ravedutti Lucio Machado",
    license='MIT',
    author_email="rafael.r.ravedutti@fau.de",
    url="https://github.com/rafaelravedutti/pairs",
    install_requires=[],
    packages=['pairs'] + [f"pairs.{mod}" for mod in modules],
    package_dir={'pairs': 'src/pairs'},
    package_data={'pairs.runtime': ['runtime/*.hpp']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/rafaelravedutti/pairs",
        "Documentation": "https://github.com/rafaelravedutti/pairs",
        "Source Code": "https://github.com/rafaelravedutti/pairs",
    },
    extras_require={},
    tests_require=[],
    python_requires=">=3.6",
)
