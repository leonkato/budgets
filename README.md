
### Repo contents

If this looks interesting, I recommend first looking at the docs directory.  There you will find a set of slides describing this project in brief.  There is also a report detailing the work and results.

Kudos to DrivenData.com for hosting the data and competition and ERS for the nice problem.

The docs directory also contains a tutorial notebook describing (with hands-on examples) different multi-target classification schemes and attempting to address some of the confusing terminology in this space.  Enjoy.

M.


### Notebooks,  files and directories

* __*data/*__  :: training data and holdout set

* __*dc_course*__  :: original version of tutorial on Drivendata competition.

* __*docs/*__  :: Project report, presentation, proposal, etc.

* __*EDA_ddbpfe*__ :: Exploratory Data Analysis of Box Plots for Education data.
    
* __*ensemble*__ :: Make an ensemble from 2 submissions, best log reg so far and default RF.

* __*feature_importances*__ :: A look at feature importance - what are the most predictive tokens?

* __*first_models*__ :: Like dc_course, but slimmed down for faster reading.

* __*first_models_metrics*__ :: Like the above but this version fits the models and saves the probability predictions and y values for train and test to disk. 

* __*fmm_out*__ :: Saved probabilty predictions from first_models_metrics.  Used by fm_standard_metrics

* __*fm_standard_metrics*__ ::  Calculate F1, accuracy, log loss, and ROC_AUC for all targets separately for each DD model separately.  Summarize.
    
* __*flat_to_labels*__ :: Shows the development of flat_to_labels, a utility to turn raw probability output from OneVsRest into properly normalized probabilities (for log loss, etc.) and label outputs (for accuracy, F1, confusion matrix, etc).

* __*mod_200/*__ :: Mod4 with my feature interaction scheme on 200 best features and regularization. 

* __*mod_400/*__ :: Mod4 with my feature interaction scheme on 400 best features and regularization. 

* __*mod_1000/*__ :: Mod4 with my feature interaction scheme on 1000 best features and regularization - these take a day to run. 

* __*mod3_1/*__  :: Best model with various tweaks and regularization (and bug fixes/workarounds)

* __*mod0_multiple_ways*__ :: Explores 2 ways to use classifiers for multi-target, multi-class:

    1. One-hot encode targets; use sklean.multilabel to drive 104 binary classifiers across this input.
    2. Use logistic regression in a multiclass fashion directly on the input (unencoded). Uses 9 different classifiers (one for each target).

* __*mod_04_99*__ :: Mod4 (all tutorial features), text data only, 1600 features.  A final attempt to get something out out DrivenData feature interaction scheme (nothing there).

* __*model_deltas*__  :: spell out each change in the DrivenData models.  Preliminary to one_at_a_time notebooks.

* __*model_out/*__  :: place to save output
 
* __*model_store/*__  :: place to save models

* __*multiclass_classifiers_examples/multioutput_classifiers_examples notebooks*__ :: These two files explore classifier probability outputs.  They differ in the way the targets are represented.  

    In multiclass, target is nd.array, n_samples by 1, containing L different labels. Probability output is nd.array (n_samples, L).  In multioutput, target is nd.array, n_samples by n_outputs (each with its own labels).  Probability output is list of array (n_samples, n_labels), one element per output.
    
* __*my_pipe*__ :: New feature interaction scheme: get just the interactions of the best features then combine with all original features.  Development.

* __*one_at_a_time_part_1.ipynb*__ :: Go through the first 2 models adding one feature engineering change at a time and see how the change impacts log loss and standard metrics. 
  
* __*one_at_a_time_part_2.ipynb*__ :: Continue adding one change at a time.

* __*one_at_a_time_p1_hv*__ ::  HashingVectorizer experiments with part 1 models.

* __*one_at_a_time_p2_scale*__ ::  Scaling experiments with part 2 models.

* __*one_at_a_time_mod4*__ :: Mod4, incremental changes.

* __*rf/*__ :: Experiments with RandomForestClassifier

* __*python/*__ ::     Utility code
    * __flat_to_labels__ - take either probabilities or y values (as n_samples by 104) and transform to 9 columns of string labels (like original data)
    * __multilabel__ - split matrix ensuring all splits have at least *n* of every label 
    * __plot_confusion_matrix__ - make nice picture of cm
    * __sparse_interactions__ - compute cartesian product of (sparse) feature matrix
    
 
    
