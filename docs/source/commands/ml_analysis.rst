ml-analysis
===========

The ``ml-analysis`` command performs machine learning model training and evaluation on processed molecular descriptor data. This command is ideal for users who already have calculated descriptors and want to focus on the machine learning aspects of the analysis.

Synopsis
--------

.. code-block:: bash

   surfacia ml-analysis [OPTIONS] -i INPUT_FILE

Description
-----------

The ml-analysis command provides comprehensive machine learning capabilities for molecular property prediction:

- **Model Training**: Supports multiple algorithms (Random Forest, XGBoost, SVM, etc.)
- **Cross-Validation**: Robust model evaluation with k-fold cross-validation
- **Feature Selection**: Intelligent feature selection maintaining predictive power
- **Performance Metrics**: Comprehensive evaluation metrics and visualizations
- **SHAP Integration**: Automatic SHAP value calculation for interpretability

This command is particularly useful when you have pre-calculated molecular descriptors and want to:

- Train predictive models quickly
- Compare different machine learning algorithms
- Perform feature selection and optimization
- Generate interpretable results with SHAP analysis

Options
-------

**Required Parameters**

.. option:: -i, --input PATH

   Input CSV file containing molecular descriptors. Must include:
   
   - ``Sample Name``: Unique identifier for each molecule
   - Descriptor columns: Numerical features for machine learning
   - Optional: Target property column for supervised learning

**Analysis Configuration**

.. option:: --test-samples TEXT

   Comma-separated list of sample indices or names to use as test set.
   
   **Examples**:
   
   - ``"1,2,3"`` - Use samples 1, 2, and 3 as test set
   - ``"caffeine,aspirin"`` - Use named samples as test set
   - ``"1-5,10,15-20"`` - Range notation supported

.. option:: --target-property TEXT

   Target property column name for supervised learning. If not provided, unsupervised analysis is performed.

.. option:: --classification

   Treat the target property as a classification problem rather than regression.

**Model Configuration**

.. option:: --algorithm TEXT

   Machine learning algorithm to use.
   
   **Options**:
   
   - ``rf`` - Random Forest (default)
   - ``xgb`` - XGBoost
   - ``svm`` - Support Vector Machine
   - ``lr`` - Linear Regression
   - ``knn`` - K-Nearest Neighbors

.. option:: --cv-folds INTEGER

   Number of cross-validation folds.
   
   **Default**: 5

.. option:: --random-state INTEGER

   Random seed for reproducible results.
   
   **Default**: 42

**Feature Selection**

.. option:: --feature-selection

   Enable automatic feature selection to reduce dimensionality and improve interpretability.

.. option:: --max-features INTEGER

   Maximum number of features to select. If not specified, uses automatic selection.

.. option:: --selection-method TEXT

   Feature selection method.
   
   **Options**:
   
   - ``stepwise`` - Stepwise selection (default)
   - ``lasso`` - LASSO regularization
   - ``rfe`` - Recursive Feature Elimination
   - ``mutual_info`` - Mutual Information

**Performance Options**

.. option:: --n-jobs INTEGER

   Number of parallel jobs for model training.
   
   **Default**: -1 (use all available cores)

.. option:: --verbose

   Enable detailed logging output.

**Output Options**

.. option:: -o, --output PATH

   Output directory for results.
   
   **Default**: ``ML_Analysis_YYYYMMDD_HHMMSS/``

Examples
--------

**Basic Regression Analysis**

.. code-block:: bash

   # Simple regression with default Random Forest
   surfacia ml-analysis -i descriptors.csv --target-property "LogP" --test-samples "1,2,3"

**Classification Analysis**

.. code-block:: bash

   # Binary classification
   surfacia ml-analysis -i descriptors.csv --target-property "Active" --classification --test-samples "10,20,30"

**Algorithm Comparison**

.. code-block:: bash

   # Compare different algorithms
   surfacia ml-analysis -i descriptors.csv --target-property "LogP" --algorithm xgb --cv-folds 10
   surfacia ml-analysis -i descriptors.csv --target-property "LogP" --algorithm svm --cv-folds 10

**Feature Selection**

.. code-block:: bash

   # Enable feature selection with maximum 20 features
   surfacia ml-analysis -i descriptors.csv --target-property "LogP" --feature-selection --max-features 20

**Advanced Configuration**

.. code-block:: bash

   # Full configuration with custom parameters
   surfacia ml-analysis -i descriptors.csv \
     --target-property "Solubility" \
     --algorithm xgb \
     --cv-folds 10 \
     --feature-selection \
     --selection-method stepwise \
     --test-samples "1-10" \
     --random-state 123 \
     --verbose

Input File Format
-----------------

The input file should contain calculated molecular descriptors:

**Required Columns**

.. csv-table::
   :header: "Column", "Description", "Example"
   :widths: 20, 50, 30

   "Sample Name", "Unique identifier", "caffeine"

**Descriptor Columns**

The file should contain numerical descriptor columns such as:

.. csv-table::
   :header: "Column", "Description", "Example"
   :widths: 30, 50, 20

   "Atom Number", "Number of atoms", "24"
   "Molecule Weight", "Molecular weight (Da)", "194.19"
   "HOMO", "HOMO energy (a.u.)", "-0.25"
   "LUMO", "LUMO energy (a.u.)", "-0.05"
   "ALIE_min", "Min ALIE value (eV)", "8.5"
   "ESP_max", "Max ESP value (kcal/mol)", "45.2"

**Optional Target Property**

.. csv-table::
   :header: "Column", "Description", "Example"
   :widths: 20, 50, 30

   "LogP", "Target property (regression)", "1.23"
   "Active", "Target property (classification)", "1"

**Example Input File**

.. code-block:: text

   Sample Name,Atom Number,Molecule Weight,HOMO,LUMO,ALIE_min,ESP_max,LogP
   caffeine,24,194.19,-0.25,-0.05,8.5,45.2,-0.07
   aspirin,21,180.16,-0.28,-0.03,8.8,52.1,1.19
   ibuprofen,31,206.28,-0.22,-0.01,8.2,38.5,3.97

Output Files
------------

The ml-analysis command generates comprehensive results:

**Primary Results**

.. code-block:: text

   ML_Analysis_20241201_143022/
   ├── model_performance.png           # Performance visualization
   ├── feature_importance.csv          # Feature importance ranking
   ├── cross_validation_results.csv    # CV scores and metrics
   ├── predictions.csv                 # Model predictions
   ├── shap_values.csv                 # SHAP values for interpretability
   └── model_summary.txt               # Analysis summary

**Performance Visualizations**

.. code-block:: text

   ├── plots/
   │   ├── actual_vs_predicted.png     # Regression: actual vs predicted
   │   ├── residuals_plot.png          # Regression: residuals analysis
   │   ├── confusion_matrix.png        # Classification: confusion matrix
   │   ├── roc_curve.png              # Classification: ROC curve
   │   ├── feature_importance.png      # Feature importance plot
   │   └── shap_summary.png           # SHAP summary plot

**Model Files**

.. code-block:: text

   ├── models/
   │   ├── trained_model.pkl           # Serialized trained model
   │   ├── feature_selector.pkl        # Feature selection transformer
   │   └── preprocessing_pipeline.pkl  # Data preprocessing pipeline

Performance Metrics
-------------------

**Regression Metrics**

- **R²**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

**Classification Metrics**

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for each class
- **Recall**: Recall for each class
- **F1-Score**: F1-score for each class
- **AUC-ROC**: Area Under ROC Curve

**Cross-Validation**

All metrics are reported with:

- **Mean**: Average across CV folds
- **Standard Deviation**: Variability across folds
- **95% Confidence Interval**: Statistical confidence bounds

Algorithm Details
-----------------

**Random Forest (rf)**

.. code-block:: python

   # Default parameters
   RandomForestRegressor(
       n_estimators=100,
       max_depth=None,
       min_samples_split=2,
       min_samples_leaf=1,
       random_state=42
   )

**XGBoost (xgb)**

.. code-block:: python

   # Default parameters
   XGBRegressor(
       n_estimators=100,
       max_depth=6,
       learning_rate=0.1,
       subsample=1.0,
       random_state=42
   )

**Support Vector Machine (svm)**

.. code-block:: python

   # Default parameters
   SVR(
       kernel='rbf',
       C=1.0,
       gamma='scale',
       epsilon=0.1
   )

Feature Selection Methods
--------------------------

**Stepwise Selection**

- Forward/backward selection based on statistical significance
- Maintains model interpretability
- Balances performance and simplicity

**LASSO Regularization**

- L1 regularization for automatic feature selection
- Produces sparse models
- Good for high-dimensional data

**Recursive Feature Elimination (RFE)**

- Iteratively removes least important features
- Works with any estimator
- Provides feature ranking

**Mutual Information**

- Measures dependency between features and target
- Captures non-linear relationships
- Good for complex feature interactions

Best Practices
--------------

**Data Preparation**

1. **Clean Data**: Remove missing values and outliers
2. **Feature Scaling**: Ensure features are on similar scales
3. **Correlation Check**: Remove highly correlated features
4. **Validation**: Use appropriate train/test splits

**Model Selection**

1. **Start Simple**: Begin with Random Forest for baseline
2. **Compare Algorithms**: Test multiple algorithms
3. **Cross-Validation**: Use sufficient CV folds (5-10)
4. **Feature Selection**: Enable for better interpretability

**Performance Evaluation**

1. **Multiple Metrics**: Don't rely on single metric
2. **Statistical Significance**: Check confidence intervals
3. **Overfitting**: Monitor train vs validation performance
4. **Domain Knowledge**: Validate results with chemical intuition

**Interpretability**

1. **SHAP Analysis**: Always examine SHAP values
2. **Feature Importance**: Understand key molecular features
3. **Chemical Meaning**: Connect results to chemical principles
4. **Visualization**: Use plots for better understanding

Troubleshooting
---------------

**Common Issues**

.. admonition:: Poor model performance
   :class: warning

   **Symptoms**: Low R² or accuracy scores
   
   **Solutions**:
   
   - Check data quality and preprocessing
   - Try different algorithms
   - Enable feature selection
   - Increase cross-validation folds

.. admonition:: Overfitting
   :class: warning

   **Symptoms**: High training score, low validation score
   
   **Solutions**:
   
   - Reduce model complexity
   - Enable regularization
   - Use more training data
   - Apply feature selection

.. admonition:: Memory issues
   :class: warning

   **Symptoms**: ``MemoryError`` during training
   
   **Solutions**:
   
   - Reduce number of features
   - Use simpler algorithms
   - Decrease n_jobs parameter
   - Process data in smaller batches

Integration with Other Commands
--------------------------------

**From Workflow**

.. code-block:: bash

   # Extract descriptors from workflow results
   surfacia ml-analysis -i Surfacia_3.0_*/FinalFull*.csv --target-property "LogP"

**To SHAP Visualization**

.. code-block:: bash

   # Use ML results for detailed SHAP analysis
   surfacia shap-viz -i ML_Analysis_*/predictions.csv --api-key YOUR_KEY

**Iterative Improvement**

.. code-block:: bash

   # Compare different feature selection methods
   surfacia ml-analysis -i data.csv --feature-selection --selection-method stepwise
   surfacia ml-analysis -i data.csv --feature-selection --selection-method lasso
   surfacia ml-analysis -i data.csv --feature-selection --selection-method rfe

See Also
--------

- :doc:`workflow` - Complete analysis pipeline
- :doc:`shap_viz` - Interpretable visualization
- :doc:`utilities` - Supporting tools
- :doc:`../getting_started/quick_start` - Quick start guide