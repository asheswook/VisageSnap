import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

requirements = [
    'numpy',
    'scikit-learn',
    'dlib',
    'face_recognition',
    'pickle',
]

setuptools.setup(
    name='face_classification',
    version='0.1',
    author='Jaewook Lee',
    author_email='me@jwlee.xyz',
    description='Face Classification package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/asheswook/face-classification',
    project_urls={
        'Bug Tracker': 'https://github.com/asheswook/face-classification/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'face_classification'},
    package=setuptools.find_packages(where='face_classification'),
    python_requires='>=3.7',
    install_requires=requirements,
)