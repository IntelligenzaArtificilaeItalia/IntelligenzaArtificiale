import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="intelligenzaartificiale",
    version="0.0.0.37",
    author="intelligenzaartificialeitalia.net",
    author_email="ceo@intelligenzaartificialeitalia.net",
    description="Intelligenza Artificiale la libreria python italiana dedicata all'I.A.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    #project_urls={
        #"Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    #},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        'pydataset==0.2.0',
        'texthero==1.1.0',
        'dtale==2.2.0',
        'numpy',
        'shap==0.40.0',
        'matplotlib',
        'pandas_profiling==3.1.0',
        'wordcloud==1.8.1',
        'seaborn==0.11.2',
        'pandas',
        'datatable==1.0.0',
        'scikit_learn',
        'Jinja2==3.1.1',
        'Flask==2.0.3'
    ],
    python_requires=">=3.6",
)