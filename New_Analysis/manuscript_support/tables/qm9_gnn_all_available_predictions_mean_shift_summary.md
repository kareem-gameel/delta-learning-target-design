# QM9 GNN Transfer Metrics: Mean-Shifted Summary

Rows are available local QM9 predictions evaluated against `DFT_ref,BoE`; values use oracle mean-shift calibration. Legacy PRE6p and `DFT_ref`-scale rows are retained in the full CSV but omitted here.

| gnn_model   | trained_target_or_stack                     | target_source   |      n |   mae_Ha |   rmse_Ha |       r2 |   kendall_tau |   shift_Ha |
|:------------|:--------------------------------------------|:----------------|-------:|---------:|----------:|---------:|--------------:|-----------:|
| GemNet      | xTB_ref,BoE + Delta^xTB_ref,BoE             | linear          | 130807 |  0.00615 |   0.00858 |  0.97229 |       0.89909 |   -0.02152 |
| GemNet      | DFT_ref,BoE                                 | direct          | 130807 |  0.01557 |   0.02945 |  0.67321 |       0.744   |    0.2451  |
| GemNet      | BoB_ref,BoE + Delta^BoB_ref,BoE             | linear          | 128988 |  0.01888 |   0.03319 |  0.57603 |       0.68115 |    0.26206 |
| GemNet      | xTB_ref,BoE                                 | direct          | 130807 |  0.02865 |   0.04271 |  0.31267 |       0.4819  |    0.24897 |
| GemNet      | BoB_MLP_ref,BoE + Delta^MLP_BoB_ref,BoE     | MLP             | 130807 |  0.03015 |   0.04287 |  0.30743 |       0.5128  |    0.2716  |
| GemNet      | PRE_ref,BoE + Delta^PRE_ref,BoE             | linear          | 130807 |  0.03316 |   0.06308 | -0.49922 |       0.67511 |    0.21205 |
| GemNet      | PRE+A_ref,BoE + Delta^PRE+A_ref,BoE         | linear          | 128988 |  0.03478 |   0.07329 | -1.06668 |       0.73271 |    0.24798 |
| GemNet      | PRE_MLP_ref,BoE + Delta^MLP_PRE_ref,BoE     | MLP             | 128988 |  0.05981 |   0.10747 | -3.44445 |       0.61011 |    0.21152 |
| GemNet      | PRE+A_MLP_ref,BoE + Delta^MLP_PRE+A_ref,BoE | MLP             | 128988 |  0.06997 |   0.12839 | -5.34318 |       0.71951 |    0.19115 |
| SchNet      | xTB_ref,BoE + Delta^xTB_ref,BoE             | linear          | 130807 |  0.00703 |   0.00992 |  0.96294 |       0.89673 |   -0.0262  |
| SchNet      | DFT_ref,BoE                                 | direct          | 130807 |  0.01379 |   0.02645 |  0.73631 |       0.77899 |    0.25194 |
| SchNet      | xTB_ref,BoE                                 | direct          | 130807 |  0.02249 |   0.03532 |  0.52992 |       0.59677 |    0.25132 |
| SchNet      | BoB_MLP_ref,BoE + Delta^MLP_BoB_ref,BoE     | MLP             | 130807 |  0.02788 |   0.03966 |  0.40729 |       0.53506 |    0.27844 |
| SchNet      | PRE+A_ref,BoE + Delta^PRE+A_ref,BoE         | linear          | 128988 |  0.04042 |   0.07918 | -1.41229 |       0.72301 |    0.24278 |
| SchNet      | BoB_ref,BoE + Delta^BoB_ref,BoE             | linear          | 128988 |  0.04403 |   0.05938 | -0.3566  |       0.51273 |    0.17217 |
| SchNet      | PRE_ref,BoE + Delta^PRE_ref,BoE             | linear          | 130807 |  0.05327 |   0.0752  | -1.13119 |       0.53472 |    0.16966 |
| SchNet      | PRE_MLP_ref,BoE + Delta^MLP_PRE_ref,BoE     | MLP             | 128988 |  0.06015 |   0.10706 | -3.41014 |       0.59088 |    0.21436 |
| SchNet      | PRE+A_MLP_ref,BoE + Delta^MLP_PRE+A_ref,BoE | MLP             | 128988 |  0.07036 |   0.12873 | -5.37671 |       0.72028 |    0.19068 |
