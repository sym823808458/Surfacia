from setuptools import setup, find_packages

setup(
    name='surfacia',
    version='2.2.1',
    description='SURF Atomic Chemical Interaction Analyzer',
    author='Yuming Su',
    author_email='823808458@qq.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'shap',
        'xgboost',
        'scikit-learn',
        'joblib',
        # 'openbabel', # Removed because OpenBabel is installed via conda
    ],
    entry_points={
        'console_scripts': [
            'surfacia=scripts.surfacia_main:main',  # Adjust based on your actual script
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.7',  # Adjust if needed
        'License :: OSI Approved :: MIT License',  # Adjust based on your LICENSE
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
