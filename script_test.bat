call conda activate mvecf
call python labcode_efficient_test.py --temp 1 --ssl_reg 1e-6 --lr 1e-2 --hyperNum 128 --gnn_layer 2 --data_type "CRSP" --target_year 2015 --save_path "./candidate1"
call python labcode_efficient_test.py --temp 1 --ssl_reg 1e-6 --lr 5e-2 --hyperNum 128 --gnn_layer 2 --data_type "CRSP" --target_year 2015 --save_path "./candidate2"
call python labcode_efficient_test.py --temp 1 --ssl_reg 1e-6 --lr 1e-1 --hyperNum 128 --gnn_layer 2 --data_type "CRSP" --target_year 2015 --save_path "./candidate3"
