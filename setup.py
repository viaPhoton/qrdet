from setuptools import setup, find_namespace_packages

setup(
    name='qrdet',
    version='2.4.1',
    author_email='eric@ericcanas.com',
    author='Eric Canas; viaPhoton customization',
    url='https://github.com/viaPhoton/qrdet',
    description='Robust QR Detector based on YOLOv8',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(),
    # expose qreader.py as the unique module
    license='MIT',
    install_requires=[
        'ultralytics==8.2.2',
        'onnx',
        'onnxruntime',
        'quadrilateral-fitter',
        'boto3',
        'numpy',
        'requests',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Topic :: Multimedia :: Graphics',
        'Typing :: Typed',
    ],
)