# Low-Resource Text Classification Framework

Introduced in [Ein-dor et al. (2020)](#reference), this is a framework for experimenting with text classification tasks.
The focus is on low-resource scenarios, and examining how active learning (AL) can be used in combination with
classification models.

The framework includes a selection of labeled datasets, machine learning models and active learning strategies (see 
[Built-in Implementations](#built-in-implementations) below), and can be easily adapted for additional setups and
scenarios.


*NOTE:* the code will be available by November 20th 2020.


**Table of contents**

[Installation](#installation)

[Running active learning experiments](#running-active-learning-experiments)

[Adapting to additional scenarios](#adapting-to-additional-scenarios):
* [Implementing a new machine learning model](#implementing-a-new-machine-learning-model)
* [Implementing a new active learning strategy](#implementing-a-new-active-learning-strategy)
* [Adding a new dataset](#adding-a-new-dataset)

[Built-in Implementations](#built-in-implementations)

[Reference](#reference)

[License](#license)

## Installation
1. Clone the repository locally: 

   `git clone https://github.com/IBM/low-resource-text-classification-framework`
2. Install the project dependencies: `pip install -r lrtc_lib/requirements.txt`
3. Run the shell script `lrtc_lib/download_and_prepare_datasets.sh`.
This script downloads the [datasets with built-in support](#built-in-implementations).

## Running active learning experiments
The `ExperimentRunner` class enables running experiments in the vein of [Ein-dor et al. (2020)](#reference),
i.e. an experimental flow where an initial seed of labeled instances is used to train a model, and then several
iterations of active learning are performed. In each active learning iteration, the set of labeled instances is 
expanded with the batch examples selected by the active learning module, and a new model is trained on this larger set.

Implementations of `ExperimentRunner` vary in terms of how the initial seed of labeled instances is selected.
The three scenarios described in the paper are implemented by:
1. *ExperimentRunnerBalanced*
2. *ExperimentRunnerImbalanced*
3. *ExperimentRunnerImbalancedPractical*

The experiment flow can be performed on a custom combination of datasets, model types and active learning strategies.

The main function of each ExperimentRunner specifies all the experimental parameters. For information on all the
dataset and category names available for running experiments, run `data_access/loaded_datasets_info.py`.


## Adapting to additional scenarios

### Implementing a new machine learning model
These are the steps for integrating a new classification model:
1. Implement a new `TrainAndInferAPI`

    Machine learning models are integrated by adding a new implementation of the TrainAndInferAPI.
    The main functions are *train* and *infer*:
    
    **Train** a new model and return a unique model identifier that will be used for inference.
    ```    
    def train(self, train_data: Sequence[Mapping], dev_data: Sequence[Mapping], test_data: Sequence[Mapping], 
    train_params: dict) -> str
   ```
        
    - train_data - a list of dictionaries with at least the "text" field. Additional fields can be passed e.g.
    *[{'text': 'text1', 'additional_field': 'value1'}, {'text': 'text2', 'additional_field': 'value2'}]*
    - dev_data: can be None if not used by the implemented model
    - test_data - can be None if not used by the implemented model
    - train_params - dictionary for additional train parameters (can be None)

    **Infer** a given sequence of elements and return the results.

    ```    
    def infer(self, model_id, items_to_infer: Sequence[Mapping], infer_params: dict, use_cache=True) -> dict:
    ```    
    - model_id
    - items_to_infer: a list of dictionaries with at least the "text" field. Additional fields can be passed,
    e.g. *[{'text': 'text1', 'additional_field': 'value1'}, {'text': 'text2', 'additional_field': 'value2'}]*
    - infer_params: dictionary for additional inference parameters (can be None)
    - use_cache: save the inference results to cache. Default is True
    
    Returns a dictionary with at least the "labels" key, where the value is a list of numeric labels for each element in
    items_to_infer.
    Additional keys (with list values of the same length) can be passed,
    e.g. *{"labels": [1, 0], "gradients": [[0.24, -0.39, -0.66, 0.25], [0.14, 0.29, -0.26, 0.16]]}*

2. Specify a new ModelType in `ModelTypes`
3. Return the newly implemented TrainAndInferAPI in `TrainAndInferFactory`
4. The system assumes that active learning strategies that require special inference outputs (e.g. text embeddings)
are not supported by your new model. If your model does support this, add it to the appropriate category in `get_compatible_models`
5. Set your ModelType in one of the ExperimentRunners, and run

### Implementing a new active learning strategy
These are the steps for integrating a new active learning approach:
1. Implement a new `ActiveLearner`
   
   Active learning modules inherit from the ActiveLearner API.
   The main function to implement is *get_recommended_items_for_labeling*:
   ```
   def get_recommended_items_for_labeling(self, workspace_id: str, model_id: str, dataset_name: str,
                                           category_name: str, sample_size: int = 1) -> Sequence[TextElement]:
    
   ```    
   This function returns a batch of *sample_size* elements suggested by the active learning module for a given dataset
   and category, based on the outputs of model *model_id*.
   
   Optionally, the ActiveLearner can also implement the function `get_per_element_score`, where the active learning 
   module does not just return a batch of selected elements, but can also assign each text element with a score.

2. Specify a new ActiveLearningStrategy in `ActiveLearningStrategies`
3. Return your new ActiveLearner in `ActiveLearningFactory`
4. If the active learner requires particular outputs from the machine learning model, update `get_compatible_models` 
accordingly. For instance, if the strategy relies on model embeddings, add it to the set of embedding-based strategies.
5. Set your ActiveLearningStrategy in one of the ExperimentRunners, and run

### Adding a new dataset
These are the steps for adding a new dataset:

1. Split your dataset into 3 csv files: `train.csv`, `dev.csv`, and `test.csv`. 
   1. Each line is a text element.
   1. Each file should have at least two columns: `label` and `text`, and may have additional columns.
   1. Files are placed under `lrtc_lib/data/available_datasets/<new_dataset_name>`
1. Create a processor for the new dataset by extending `CsvProcessor` (which implements `DataProcessorAPI`)
and place it under `lrtc_lib/data_access/processors`.
   `CsvProcessor`  `__init__` function looks like this:
   
   ```    
   def __init__(self, dataset_name: str, dataset_part: DatasetPart, text_col: str = 'text',
                 label_col: str = 'label', context_col: str = None,
                 doc_id_col: str = None,
                 encoding: str = 'utf-8'):
   ``` 

    - dataset_name: the name of the processed dataset
    - dataset_part: the part - train/dev/test - of the dataset
    - text_col: the name of the column which holds the text of the TextElement. Default is `text`.
    - label_col: the name of the column which holds the label. Default is `label`.
    - context_col: the name of the column which provides context for the text, None if no context is available.
    Default is None.
    - doc_id_col: column name by which text elements should be grouped into documents.
    If None, all text elements would be put in a single dummy doc. Default is None.
    - encoding: the encoding to use to read the csv raw file(s). Default is `utf-8`.
    
    For example, here is the processor for DBPedia (which uses the default values of `CsvProcessor`):
       
   ```
    class DbpediaProcessor(CsvProcessor):

    def __init__(self, dataset_part: DatasetPart):
        super().__init__(dataset_name='dbpedia', dataset_part=dataset_part)
   ```
    
   If more flexibility is needed, implement `DataProcessorAPI` directly. 
1. Add the new processor to `data_processor_factory`. Note, in this step you define the name of the new dataset. 
1. Run `load_dataset` with the new dataset name (as defined in `data_processor_factory`) to generate dump files under 
`data/data_access_dumps` (for the documents and text elements of the dataset) and `data/oracle_access_dumps` 
(for the gold labels of the text elements).


## Built in Implementations
### Datasets
- [AG’s News](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
- [CoLA](https://nyu-mll.github.io/CoLA/)
- [ISEAR](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)*
- [Polarity](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
- [Subjectivity](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
- [TREC](https://cogcomp.seas.upenn.edu/Data/QA/QC/)
- [Wiki Attack](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release)

\* _Loading the ISEAR dataset requires installing additional dependencies before 
[running the installation script](#installation), and is only supported on Mac/Linux. Specifically, you will need to 
install [mdbtools](https://github.com/mdbtools/mdbtools) on your machine and then `pip install pandas_access`_.


### Classification models
- **ModelTypes.NB**: a Naive Bayes implementation from [scikit-learn](https://scikit-learn.org)
- **ModelTypes.HF_BERT**: A tensorflow implementation of BERT (Devlin et al. 2018) that uses the [huggingface Transformers](https://github.com/huggingface/transformers) library 

### Active learning strategies
- **RANDOM**: AL baseline, randomly sample from unlabeled data.
- **RETROSPECTIVE**: select the top scored instances by the model.
- **HARD_MINING**: a.k.a uncertainty sampling / least confidence; Lewis and Gale 1994
- **GREEDY_CORE_SET**: the greedy method from Sener and Savarese 2017
- **DAL**: Discriminative representation sampling; Gissin and Shalev-Shwartz 2019
- **PERCEPTRON_ENSEMBLE**: lightweight ensemble version of uncertainty sampling; uncertainty is determined
using an ensemble of perceptrons, which were trained over output embeddings from the original model.
- **DROPOUT_PERCEPTRON**: similar to the above, but instead of an ensemble of perceptrons, uses a single perceptron
with Monte Carlo dropout (Gal and Ghahramani, 2016)


## Reference
Liat Ein-Dor, Alon Halfon, Ariel Gera, Eyal Shnarch, Lena Dankin, Leshem Choshen, Marina Danilevsky, Ranit Aharonov, Yoav Katz and Noam Slonim (2020). Active Learning for BERT: An Empirical Study. EMNLP 2020

## License
This work is released under the Apache 2.0 license. The full text of the license can be found in [LICENSE](LICENSE).
