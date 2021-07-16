# torchcsvdataset
experimental


# i was motivated by [tf.data.experimental.CsvDataSet](http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/data/experimental/CsvDataset.html)


# CSVDataset
* pandas
  * only csv (now status)
* vaex 
  * hdf5 (now status)

# Expectation
* Reduce memory usage

# Feature
* read(batch)
* transform(not yet)
* ...

# Check Memory

* parameter 
$1 n_samples
$2 n_features
$3 batch_size

* output_folder 
./



```
./test/test.sh 1000 5 10 
```
* 
