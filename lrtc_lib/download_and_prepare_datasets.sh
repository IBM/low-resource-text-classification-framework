#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux"* ]]; then
realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}
fi

curr_dir=`dirname $0`
base_dir=`realpath "${curr_dir}"/data/`
raw_dir=`realpath "${base_dir}"/raw/`
python_path=`realpath "${curr_dir}"/../`

mkdir "$raw_dir"
cd "$raw_dir"
echo "$raw_dir"

##### polarity
if [ ! -f polarity.tar.gz ] && [ ! -d polarity_imbalanced_positive ] ; then
  echo '** Downloading polarity files **'
#  curl https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz -o polarity.tar.gz
  curl https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz -o polarity.tar.gz
fi

if  [ ! -d polarity ] ; then
  tar -xzf polarity.tar.gz
  mv rt-polaritydata polarity
  python "$base_dir"/get_by_ids.py polarity

  cp -r polarity polarity_imbalanced_positive
  python "$base_dir"/get_by_ids.py polarity_imbalanced_positive
  rm polarity.tar.gz
fi

##### subjectivity
if [ ! -f subjectivity.tar.gz ] && [ ! -d subjectivity ] ; then
  echo '** Downloading subjectivity files **'
  curl http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz -o subjectivity.tar.gz
  mkdir -p subjectivity
  tar -xzf subjectivity.tar.gz -C subjectivity
  python "$base_dir"/get_by_ids.py subjectivity

  cp -r subjectivity subjectivity_imbalanced_subjective
  python "$base_dir"/get_by_ids.py subjectivity_imbalanced_subjective
  rm subjectivity.tar.gz
fi

##### ag_news
if [ ! -f ag_news.tar.gz ] && [ ! -d ag_news ]; then
  # Download from google drive
  export fileid=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms
  export filename=ag_news.tar.gz
  ## CURL ##
  echo '** Downloading AG news files **'
  curl -L -c cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid \
       | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

  curl -L -b cookies.txt -o $filename \
       'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
  rm -f confirm.txt cookies.txt
fi
if [ ! -d ag_news_imbalanced_1 ]; then
  tar -xzf ag_news.tar.gz
  mv ag_news_csv ag_news
  python "$base_dir"/prepare_ag_news.py
  python "$base_dir"/get_by_ids.py ag_news

  cp -r ag_news ag_news_imbalanced_1
  python "$base_dir"/get_by_ids.py ag_news_imbalanced_1
  rm ag_news.tar.gz
fi

#### wiki_attack
if [ ! -d wiki_attack ]; then
  echo '** Downloading wiki attack files **'
  curl  https://ndownloader.figshare.com/articles/4054689/versions/6 -o wiki_attack.zip
  unzip wiki_attack.zip -d ./wiki_attack
  #cp $base_dir/prepare_wiki_attack.py $raw_dir/wiki_attack/prepare_wiki_attack.py
  python "$base_dir"/prepare_wiki_attack.py
  python "$base_dir"/get_by_ids.py wiki_attack
  rm wiki_attack.zip
fi

#### trec
if [ ! -d trec ]; then
  echo '** Downloading TREC files **'
  mkdir trec
  cd trec
  curl -O https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label
  curl -O https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label
  mkdir -p ../../available_datasets/trec
  cp TREC_10.label ../../available_datasets/trec/test.txt
  cd ../../available_datasets
  python "$base_dir"/prepare_trec.py
  cd "$raw_dir"
fi

#### cola
if [ ! -f "./cola/dev.csv" ]; then
  echo '** Downloading CoLA files **'
  curl https://nyu-mll.github.io/CoLA/cola_public_1.1.zip -o cola.zip
  unzip cola.zip -d ./cola
  python "$base_dir"/prepare_cola.py
  python "$base_dir"/get_by_ids.py cola
  rm cola.zip
fi


#### ISEAR
############# NOTE ISEAR requires special dependencies
# pip install pandas_access
# https://github.com/mdbtools/mdbtools
if [ ! -d isear ]; then
  echo '** Downloading ISEAR files **'
  # link redirects here https://www.unige.ch/cisa/files/9114/6719/1920/ISEAR_0.zip
  curl https://www.unige.ch/cisa/index.php/download_file/view/395/296/ -o isear.zip -L
  unzip isear.zip -d ./isear
  cd isear
  unzip isear_databank
  rm ../isear.zip
  cd "$raw_dir"
fi

if [ ! -f "./isear/isear_data.csv" ]; then
  python "$base_dir"/prepare_isear.py
  if [ -f "./isear/isear_data.csv" ]; then
    python "$base_dir"/get_by_ids.py isear
  fi
fi

if [ -f "./isear/isear_data.csv" ]; then
  cd "$python_path"
  python -c "import sys; import os; sys.path.append(os.getcwd()); os.chdir(os.path.join(os.getcwd(), 'lrtc_lib')); import lrtc_lib.data.load_dataset as loader; loader.load('isear')"
fi

cd "$python_path"
python -c "import sys; import os; sys.path.append(os.getcwd()); os.chdir(os.path.join(os.getcwd(), 'lrtc_lib'));
import lrtc_lib.data.load_dataset as loader
loader.load('polarity');
loader.load('polarity_imbalanced_positive');
loader.load('subjectivity');
loader.load('subjectivity_imbalanced_subjective')
loader.load('ag_news');
loader.load('ag_news_imbalanced_1');
loader.load('wiki_attack');
loader.load('trec');
loader.load('cola')"
