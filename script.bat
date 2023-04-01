call conda activate mvecf
for %%a in ( "CRSP" "THOMSON13f" ) do (
    for %%b in ( 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 ) do (
        call python labcode_efficient.py --temp 1 --ssl_reg 1e-6 --lr 1e-2 --gnn_layer 2 --data_type %%a --target_year %%b
    )
)
