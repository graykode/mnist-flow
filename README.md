# mnist-flow

This Project is only repository for solving [AI Engineer Party](https://www.facebook.com/groups/TensorFlowKR/permalink/971390023202056/). DataSet is MNIST to reduce GCE cost.



## Architecture

![](/Users/graykode/Desktop/architecture.png)



## Quick Start using GCE Configuration

1. Set your GCE region, Please use this command on Python2.7, beacase of gcloud cli. To use gcloud, it had been connected your GCE Account, see `gcloud info`.

   ```shell
    $ gcloud config set compute/zone asia-northeast1
   ```

2. Install Cloud Dataflow Python SDK

   ```shell
   $ pip install --upgrade google-cloud-dataflow --user
   ```
   
4. Create IAM which have roles in BigQuery and Google Storage View.

   ```shell
   $ export GOOGLE_APPLICATION_CREDENTIALS="key.json"
   ```
   
4. Preprocess `tf.keras.datasets.mnist` to be flatten in [script/make_data.py](script/make_data.py) and upload to Google Bigquery as splitting train/test dataset.

   ```shell
   $ python script/make_data.py
   $ gzip data/*.txt
   $ bq load --source_format=CSV -F":" mnist.train data/train.txt.gz \
       "key:integer,image:string,label:integer"
   $ bq load --source_format=CSV -F":" mnist.test data/test.txt.gz \
       "key:integer,image:string,label:integer"
   ```

5. Install BigQuery Client Libraries if you want to run on real time. **Be careful call one query making below traffic.  Google BigQuery price is $0.035 per GB**. So we should be cached dataset.

   - 60000 MNIST train Dataset : 151.4MB
   - 10000 MNIST test Dataset : 25.3MB

   ```python
   from google.cloud import bigquery
   client = bigquery.Client()
   query = ("SELECT image, label FROM mnist.train")
   query_job = client.query(
       query,
   )  # API request - starts the query
   for row in query_job:  # API request - fetches results
       print(row)
   ```

6. To Training Adanet, run [train_adanet.py](train_adanet.py])

   ```shell
   $ python train_adanet.py \
   			--RANDOM_SEED 42 \
   			--NUM_CLASSES 10 \
   			--TRAIN_STEPS 5000 \
   			--BATCH_SIZE 64 \
   			--LEARNING_RATE 0.001 \
   			--FEATURES_KEY "images" \
   			--ADANET_ITERATIONS 2\
   			--MODEL_DIR "./models" \
   			--SAVED_DIR "./saved_model" \
   			--logfile "log.txt"
   ```

   See saved model directory structure below:

   ```
   models
   └── 1566790420
       ├── saved_model.pb
       └── variables
           ├── variables.data-00000-of-00001
           └── variables.index
   ```

7. Create Google Storage Bucket to upload saved model folder

   ```shell
   $ PROJECT_ID=$(gcloud config list project --format "value(core.project)")
   $ BUCKET="${PROJECT_ID}-ml"
   
   # create bucket
   $ gsutil mb -c regional -l asia-northeast1 gs://${BUCKET}
   
   # upload saved model
   $ gsutil -m cp -R ./saved_model gs://${BUCKET}
   ```

8. Deploy google function

   ```shell
   $ cd gfunction
   $ gcloud functions deploy handler --runtime python37 \
   				--trigger-http --memory 2048 --region asia-northeast1
   ```

9. Test API using `curl`

   ```shell
   $ curl -F 'file=@testdata/0.png' 'your api handler server'
   output : 0
   ```

#### pylint

```shell
$ pip install pylint
$ pylint **/*.py
```

But I use pep8 supported on Pycharm.



## Author

- Tae Hwan Jung(Jeff Jung) @graykode, Kyung Hee Univ CE(Undergraduate).
- Author Email : [nlkey2022@gmail.com](mailto:nlkey2022@gmail.com)