for data_type, target_year, positive_score_cri in [
    ("CRSP", 2009, 0.147881246),
    ("CRSP", 2010, 0.144944502),
    ("CRSP", 2011, 0.144274695),
    ("CRSP", 2012, 0.195083213),
    ("CRSP", 2006, 0.25819187),
    ("CRSP", 2007, 0.196639571),
    ("CRSP", 2008, 0.18414649),
    ("CRSP", 2013, 0.321803689),
    ("CRSP", 2014, 0.373010027),
    ("CRSP", 2015, 0.362223498),
    ("THOMSON13f", 2006, 0.258347843),
    ("THOMSON13f", 2007, 0.194829434),
    ("THOMSON13f", 2008, 0.18043514),
    ("THOMSON13f", 2009, 0.147997115),
    ("THOMSON13f", 2010, 0.14446381),
    ("THOMSON13f", 2011, 0.146935004),
    ("THOMSON13f", 2012, 0.199539768),
    ("THOMSON13f", 2013, 0.364394073),
    ("THOMSON13f", 2014, 0.41655308),
    ("THOMSON13f", 2015, 0.416697949)
]:
    print(f'call python labcode_efficient.py --temp 1 --ssl_reg 1e-6 --lr 1e-2 --gnn_layer 2 --data_type "{data_type}" --target_year {target_year} --positive_score_cri {positive_score_cri}')