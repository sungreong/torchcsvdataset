python ./test/make_data.py --n_samples $1 --n_features $2
mprof run ./test/vaex_test.py --n_samples $1 --batch_size $3 
mprof plot -o ./vaex_$1_$3_plot.png --backend agg 
mprof run ./test/pandas_test.py --n_samples $1 --batch_size $3 
mprof plot -o ./pandas_$1_$3_plot.png --backend agg 
