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

# Conclusion

* pandas
  * 느리지만, 메모리는 거의 사용하지 않음을 알 수 있음. 
  * 그러나, random sampling 같은 것을 할 수가 없음
* vaex 
  * 빠르지만, 메모를 더 많이 사용하는 경향이 있음
  * 다양한 기능등을 추가적으로 사용할 수 있으나 메모리를 확인해봐야 함(TODO)
