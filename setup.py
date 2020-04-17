import setuptools

long_description = "prettierplot is a Python library that makes it easy to create high-quality, polished data visualizations with minimal code."

description = "Quickly create prettier plots"
distname = "prettierplot"
license = "MIT"
maintainer = "Tyler Peterson"
maintainer_email = "petersontylerd@gmail.com"
project_urls = {
    "bug tracker": "https://github.com/petersontylerd/prettierplot/issues",
    "source code": "https://github.com/petersontylerd/prettierplot",
}
url = "https://github.com/petersontylerd/prettierplot"
version = "0.1.2"


def setup_package():
    metadata = dict(
        name=distname,
        packages=[
            "prettierplot",
            "prettierplot.datasets",
            "prettierplot.datasets.attrition",
            "prettierplot.datasets.housing",
            "prettierplot.datasets.titanic",
            ],
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        description=description,
        keywords=["machine learning", "data science"],
        license=license,
        url=url,
        # download_url = download_url,
        project_urls=project_urls,
        version=version,
        long_description=long_description,
        include_package_data=True,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.6.1",
        install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    )

    setuptools.setup(**metadata)


if __name__ == "__main__":
    setup_package()
