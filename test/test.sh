python ./test/make_data.py --n_samples $1 --n_features $2
mprof run ./test/vaex_test.py --n_samples $1 -n_features $2 --batch_size $3 
mprof plot -o ./vaex_n_$1_bn_$3_n_$2_plot.png --backend agg 
mprof run ./test/pandas_test.py --n_samples $1 -n_features $2 --batch_size $3 
mprof plot -o ./pandas_n_$1_bn_$3_n_$2_plot.png --backend agg 
