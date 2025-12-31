Machine Learning Modules
=========================

This section documents the machine learning modules of Surfacia that provide feature selection, model training, SHAP analysis, and intelligent feature recommendations.

.. module:: surfacia.ml.chem_ml_analyzer_v2

Chemical Machine Learning Analyzer
----------------------------------

.. automodule:: surfacia.ml.chem_ml_analyzer_v2
   :members:
   :undoc-members:
   :show-inheritance:

The chemical machine learning analyzer provides comprehensive tools for training XGBoost models on molecular descriptors, performing feature selection, and generating SHAP-based interpretations.

Classes
~~~~~~~

BaseChemMLAnalyzer
~~~~~~~~~~~~~~~~~~

.. autoclass:: surfacia.ml.chem_ml_analyzer_v2.BaseChemMLAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Base class for chemical machine learning analysis. Provides core functionality for data loading, feature matrix generation, model training, and result visualization.

**Key Methods:**

- ``_load_and_prepare_data()``: Load and prepare training/test data
- ``_generate_feature_matrix()``: Generate feature matrix from merged DataFrame (to be overridden by subclasses)
- ``XGB_Fit()``: Train XGBoost model with specified parameters
- ``poolfit_optimized()``: Run optimized pooling of training runs
- ``save_comprehensive_results()``: Save model results and predictions
- ``generate_prediction_scatter()``: Generate prediction scatter plots
- ``generate_shap_plots()``: Generate SHAP analysis plots
- ``fit_various_functions()``: Fit various mathematical functions to data
- ``generate_record_file()``: Generate analysis record file

ManualFeatureAnalyzer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: surfacia.ml.chem_ml_analyzer_v2.ManualFeatureAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Analyzer for manual feature selection. Users specify which features to include in the model.

**Key Methods:**

- ``_generate_feature_matrix()``: Generate feature matrix with user-specified features
- ``run()``: Run manual feature analysis

**Usage:**

Use when you have prior knowledge about which features are important or want to test specific feature combinations.

WorkflowAnalyzer
~~~~~~~~~~~~~~~~

.. autoclass:: surfacia.ml.chem_ml_analyzer_v2.WorkflowAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Automatic workflow analyzer that performs comprehensive feature selection and model optimization through multiple analysis steps.

**Key Methods:**

- ``_generate_feature_matrix()``: Generate feature matrix for workflow analysis
- ``run()``: Run complete workflow analysis
- ``_run_baseline_analysis()``: Run baseline analysis with all features
- ``_run_stepwise_regression()``: Perform stepwise feature selection
- ``_run_shap_analysis()``: Perform SHAP-based feature analysis
- ``_intelligent_recommendation()``: Generate intelligent feature recommendations
- ``_run_final_analysis()``: Run final analysis with recommended features

**Workflow Steps:**

1. **Baseline Analysis**: Evaluate performance with all available features
2. **Feature Correlation**: Analyze correlations between features
3. **Stepwise Regression**: Multiple runs of progressive feature selection
4. **SHAP Analysis**: Interpret feature contributions
5. **Feature Recommendations**: Intelligently recommend optimal features
6. **Final Analysis**: Build optimized model with recommended features

ChemMLWorkflow
~~~~~~~~~~~~~~

.. autoclass:: surfacia.ml.chem_ml_analyzer_v2.ChemMLWorkflow
   :members:
   :undoc-members:
   :show-inheritance:

Static workflow class that provides convenient access to automatic and manual analysis modes.

**Key Methods:**

- ``run_analysis()``: Run analysis in automatic or manual mode

Examples
~~~~~~~~

**Manual feature analysis:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import ManualFeatureAnalyzer
   
   # Analyze with specific features
   analyzer = ManualFeatureAnalyzer(
       data_file="FinalFull.csv",
       manual_features=["S_ALIE_min", "C_LEAE_min", "Fun_ESP_delta"],
       generate_fitting=True
   )
   analyzer.run(
       epoch=64,
       core_num=32,
       train_test_split=0.85
   )

**Manual analysis with all features:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import ManualFeatureAnalyzer
   
   # Analyze with all features
   analyzer = ManualFeatureAnalyzer(
       data_file="FinalFull.csv",
       manual_features="Full",  # Use all available features
       generate_fitting=True
   )
   analyzer.run(epoch=64, core_num=32)

**Automatic workflow analysis:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   # Run automatic workflow
   analyzer = WorkflowAnalyzer(
       data_file="FinalFull.csv",
       max_features=5,
       n_runs=3
   )
   analyzer.run(
       epoch=64,
       core_num=32,
       train_test_split=0.85
   )

**Using ChemMLWorkflow (simplified interface):**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import ChemMLWorkflow
   
   # Automatic mode
   ChemMLWorkflow.run_analysis(
       mode="auto",
       data_file="FinalFull.csv",
       max_features=5,
       n_runs=3,
       epoch=64,
       core_num=32
   )
   
   # Manual mode
   ChemMLWorkflow.run_analysis(
       mode="manual",
       data_file="FinalFull.csv",
       manual_features=["feature1", "feature2", "feature3"],
       epoch=64,
       core_num=32
   )

**With test samples:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   analyzer = WorkflowAnalyzer(
       data_file="FinalFull.csv",
       test_sample_names=["79", "22", "82", "36", "70", "80"],
       max_features=5,
       n_runs=3
   )
   analyzer.run()

**Custom model parameters:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   # Customize XGBoost parameters
   custom_paras = {
       'max_depth': 6,
       'learning_rate': 0.05,
       'n_estimators': 200,
       'subsample': 0.8,
       'colsample_bytree': 0.8,
       'min_child_weight': 3,
       'gamma': 0.1
   }
   
   analyzer = WorkflowAnalyzer(data_file="FinalFull.csv")
   analyzer.run(paras=custom_paras)

**Accessing results:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   analyzer = WorkflowAnalyzer(data_file="FinalFull.csv")
   analyzer.run()
   
   # Results are saved automatically to organized directories
   # Access analysis results from saved files

**Advanced workflow:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   analyzer = WorkflowAnalyzer(
       data_file="FinalFull.csv",
       test_sample_names=["sample1", "sample2", "sample3"],
       max_features=8,
       n_runs=5
   )
   
   # Run with custom training parameters
   analyzer.run(
       epoch=128,              # More training iterations
       core_num=64,            # Use more CPU cores
       train_test_split=0.80,  # Different split ratio
       paras={
           'max_depth': 8,
           'learning_rate': 0.03,
           'n_estimators': 500
       }
   )

Workflow Analysis Details
~~~~~~~~~~~~~~~~~~~~~~~~

**Baseline Analysis:**

- Evaluates model performance with all available features
- Generates comprehensive performance metrics
- Creates prediction scatter plots
- Establishes baseline for comparison

**Stepwise Regression:**

- Performs multiple independent runs (default: 3)
- Each run progressively selects features
- Uses statistical criteria for feature selection
- Identifies robust features across multiple runs

**Feature Correlation:**

- Computes correlation matrix between all features
- Visualizes correlation heatmap
- Identifies highly correlated feature groups
- Helps prevent multicollinearity issues

**SHAP Analysis:**

- Computes SHAP values for each feature
- Generates SHAP summary plots
- Creates feature importance rankings
- Provides model interpretability

**Feature Recommendations:**

- Combines results from stepwise regression
- Analyzes SHAP value distributions
- Evaluates fitting quality for each feature
- Generates ranked feature recommendations

**Final Analysis:**

- Uses recommended features for final model
- Generates comprehensive performance metrics
- Creates final prediction scatter plots
- Produces detailed analysis report

Output Structure
~~~~~~~~~~~~~~~

**Workflow Mode Output:**

.. code-block:: text

   Analysis_YYYYMMDD_HHMMSS/
   ├── Baseline_Analysis/
   │   ├── prediction_scatter.png
   │   ├── data.csv
   │   ├── record_file.txt
   │   └── performance_metrics.txt
   ├── Run_1/
   │   ├── Stepwise_Results/
   │   ├── Final_Manual_Analysis/
   │   └── summary.txt
   ├── Run_2/
   │   └── ...
   ├── Run_3/
   │   └── ...
   ├── feature_recommendations.csv
   ├── workflow_summary.txt
   └── final_report.pdf

**Manual Mode Output:**

.. code-block:: text

   Manual_Feature_Analysis_YYYYMMDD_HHMMSS/
   ├── prediction_scatter.png
   ├── Training_Set_Detailed.csv
   ├── Test_Set_Detailed.csv (if test samples provided)
   ├── record_file.txt
   ├── SHAP_Plots/
   │   ├── feature1_SHAP.png
   │   ├── feature2_SHAP.png
   │   └── ...
   ├── SHAP_Raw_Data/
   │   ├── feature1_shap_data.csv
   │   ├── feature2_shap_data.csv
   │   └── ...
   └── performance_metrics.txt

Performance Metrics
~~~~~~~~~~~~~~~~~~~

**Cross-Validation Metrics:**

- **MSE**: Mean Squared Error with standard deviation
- **MAE**: Mean Absolute Error with standard deviation
- **R²**: Coefficient of Determination with standard deviation

**Test Set Metrics (if applicable):**

- **Test MSE**: Mean squared error on test set
- **Test MAE**: Mean absolute error on test set
- **Test R²**: R² score on test set

**Feature Rankings:**

- **SHAP Importance**: Average absolute SHAP values
- **Frequency**: How often features appear in top selections
- **Stability**: Consistency across multiple runs

Model Parameters
~~~~~~~~~~~~~~~

**Default XGBoost Parameters:**

.. code-block:: python

   default_paras = {
       'max_depth': 6,
       'learning_rate': 0.1,
       'n_estimators': 100,
       'subsample': 0.8,
       'colsample_bytree': 0.8,
       'min_child_weight': 1,
       'gamma': 0,
       'reg_alpha': 0,
       'reg_lambda': 1
   }

**Training Parameters:**

- ``epoch``: Number of training iterations (default: 64)
- ``train_test_split``: Fraction for training (default: 0.85)
- ``core_num``: Number of CPU cores for parallel processing (default: 32)

**Cross-Validation:**

- Uses 5-fold cross-validation by default
- Stratified splitting for classification tasks
- Preserves target distribution in splits

Feature Selection
~~~~~~~~~~~~~~~~

**Stepwise Regression Strategy:**

1. Start with no features
2. Add features one at a time
3. Evaluate model performance
4. Keep features that improve performance
5. Repeat until reaching max_features

**Evaluation Criteria:**

- Cross-validation R² score
- Feature importance (SHAP values)
- Fitting quality (mathematical relationships)
- Feature stability across runs

**SHAP-Based Selection:**

- Compute SHAP values for all features
- Rank by mean absolute SHAP value
- Select top features based on rankings
- Validate selections through cross-validation

**Fitting Analysis:**

For each feature, the analyzer fits various mathematical functions:

- Linear: ``y = ax + b``
- Polynomial: ``y = ax² + bx + c``
- Exponential: ``y = a·exp(bx) + c``
- Logarithmic: ``y = a·ln(x) + b``

Selects the best fit based on R² score.

NaN Handling
~~~~~~~~~~~~

The analyzer provides two strategies for handling missing values:

**drop_columns (default):**

- Removes features with any NaN values
- Ensures clean feature matrix
- May lose information but maintains reliability

**drop_rows:**

- Removes samples with NaN values
- Preserves all features
- Reduces training sample size

Usage:

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   analyzer = WorkflowAnalyzer(
       data_file="FinalFull.csv",
       nan_handling='drop_columns'  # or 'drop_rows'
   )
   analyzer.run()

Advanced Usage
~~~~~~~~~~~~~~

**Custom feature preprocessing:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import BaseChemMLAnalyzer
   
   class CustomAnalyzer(BaseChemMLAnalyzer):
       def _generate_feature_matrix(self, merged_df):
           # Custom feature preprocessing
           # Apply transformations, scaling, etc.
           feature_df = super()._generate_feature_matrix(merged_df)
           
           # Custom preprocessing
           feature_df = feature_df.fillna(feature_df.mean())
           feature_df = (feature_df - feature_df.mean()) / feature_df.std()
           
           return feature_df

**Ensemble feature selection:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   # Run multiple analyses with different parameters
   results = []
   for max_features in [3, 5, 8, 10]:
       analyzer = WorkflowAnalyzer(
           data_file="FinalFull.csv",
           max_features=max_features,
           n_runs=5
       )
       analyzer.run()
       results.append(analyzer)

**Hyperparameter tuning:**

.. code-block:: python

   from surfacia.ml.chem_ml_analyzer_v2 import WorkflowAnalyzer
   
   # Grid search over hyperparameters
   param_grid = {
       'max_depth': [4, 6, 8],
       'learning_rate': [0.05, 0.1, 0.15],
       'n_estimators': [100, 200, 300]
   }
   
   best_score = float('-inf')
   best_params = None
   
   for params in param_grid:
       analyzer = WorkflowAnalyzer(data_file="FinalFull.csv")
       analyzer.run(paras=params, epoch=32, core_num=16)
       # Evaluate results and update best_params

**Result aggregation:**

.. code-block:: python

   import pandas as pd
   
   # Load and aggregate results from multiple runs
   feature_recommendations = []
   
   for run_id in range(1, 4):
       df = pd.read_csv(f'Run_{run_id}/feature_recommendations.csv')
       feature_recommendations.append(df)
   
   # Aggregate across runs
   combined = pd.concat(feature_recommendations)
   avg_importance = combined.groupby('feature')['importance'].mean()
   avg_importance.sort_values(ascending=False, inplace=True)

Best Practices
~~~~~~~~~~~~~~

**1. Start with automatic workflow:**

For most use cases, the automatic workflow mode provides the best balance between automation and insight:

.. code-block:: python

   analyzer = WorkflowAnalyzer(data_file="FinalFull.csv")
   analyzer.run()

**2. Use sufficient training data:**

Ensure you have adequate training samples:

- Minimum: 20 samples
- Recommended: 50+ samples
- Optimal: 100+ samples

**3. Adjust max_features appropriately:**

Based on your dataset size:

- Small datasets (<50 samples): max_features=3-5
- Medium datasets (50-200 samples): max_features=5-10
- Large datasets (>200 samples): max_features=10-20

**4. Increase n_runs for robustness:**

More runs provide more stable feature selections:

- Quick analysis: n_runs=3
- Robust analysis: n_runs=5
- Publication-quality: n_runs=10

**5. Validate with test set:**

Always include test samples if possible:

.. code-block:: python

   analyzer = WorkflowAnalyzer(
       data_file="FinalFull.csv",
       test_sample_names=["79", "22", "82"]
   )
   analyzer.run()

**6. Review feature recommendations:**

The intelligent recommendations combine multiple criteria:

- SHAP importance
- Selection frequency
- Fitting quality

**7. Examine fitting plots:**

Fitting plots reveal mathematical relationships:

- Linear: Proportional relationship
- Exponential: Exponential growth/decay
- Logarithmic: Diminishing returns
- Polynomial: Complex relationship

**8. Monitor cross-validation metrics:**

Cross-validation R² should be:

- Excellent: R² > 0.9
- Good: R² > 0.8
- Acceptable: R² > 0.7
- Needs improvement: R² < 0.7

**9. Check feature correlations:**

High correlation (>0.9) between features may indicate:

- Redundant features
- Multicollinearity issues
- Need for feature engineering

**10. Use manual mode for expert analysis:**

When you have domain knowledge about specific features:

.. code-block:: python

   analyzer = ManualFeatureAnalyzer(
       data_file="FinalFull.csv",
       manual_features=["your_important_features"]
   )
   analyzer.run()

Integration with Workflow
~~~~~~~~~~~~~~~~~~~~~~~~

The ML analyzer integrates seamlessly with the main workflow:

.. code-block:: python

   from surfacia.core.workflow import SurfaciaWorkflow
   
   workflow = SurfaciaWorkflow("molecules.csv")
   workflow.run_full_workflow(
       max_features=5,
       stepreg_runs=3,
       epoch=64,
       cores=32
   )

The workflow automatically:

1. Extracts features from surface analysis
2. Runs ML analysis with specified parameters
3. Generates SHAP visualizations
4. Produces interactive web interface

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**Parallel Processing:**

The analyzer utilizes multiple CPU cores for efficient computation:

.. code-block:: python

   analyzer.run(core_num=64)  # Use 64 cores

**Memory Management:**

For large datasets:

- Increase system memory
- Use ``nan_handling='drop_columns'`` to reduce memory
- Process features in batches if needed

**Computation Time:**

Estimated times for different dataset sizes:

- Small (50 samples, 10 features): ~5-10 minutes
- Medium (100 samples, 20 features): ~20-30 minutes
- Large (200+ samples, 50+ features): ~1-2 hours

Troubleshooting
~~~~~~~~~~~~~~~

**Issue: Poor model performance**

**Solutions:**
1. Check for sufficient training data
2. Verify feature quality and relevance
3. Try different hyperparameters
4. Examine feature correlations
5. Consider feature engineering

**Issue: NaN values causing errors**

**Solutions:**
1. Use ``nan_handling='drop_columns'`` to remove problematic features
2. Use ``nan_handling='drop_rows'`` to remove problematic samples
3. Investigate why NaN values exist

**Issue: Overfitting**

**Symptoms:**
- High training R², low test R²
- Large gap between CV and test performance

**Solutions:**
1. Increase regularization in XGBoost parameters
2. Reduce number of features
3. Increase training data
4. Use more conservative model parameters

**Issue: Unstable feature selection**

**Symptoms:**
- Different features selected in each run
- High variance in feature importance

**Solutions:**
1. Increase ``n_runs`` parameter
2. Use larger training datasets
3. Increase ``epoch`` for better convergence
4. Consider manual feature selection

**Issue: Memory errors**

**Solutions:**
1. Reduce number of features
2. Use ``nan_handling='drop_columns'``
3. Reduce ``core_num`` parameter
4. Process data in smaller batches

These machine learning modules provide powerful tools for analyzing molecular surface descriptors, selecting important features, and building interpretable predictive models with SHAP-based explanations.
