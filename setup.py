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
    name='FaceFlow',
    version='0.1',
    author='Jaewook Lee',
    author_email='me@jwlee.xyz',
    description='Face Classification package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/asheswook/FaceFlow',
    project_urls={
        'Bug Tracker': 'https://github.com/asheswook/FaceFlow/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'FaceFlow'},
    package=setuptools.find_packages(where='FaceFlow'),
    python_requires='>=3.7',
    install_requires=requirements,
)