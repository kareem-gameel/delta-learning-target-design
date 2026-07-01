# Delta target linear vs MLP summary

Scale columns are computed on the tmQM validation rows after applying `New_Analysis/pre_predict_targets/outputs/pre_shared_outlier_mask.csv`. QM9 columns are oracle mean-shifted MAE against QM9 `DFT_ref,BoE`.

| target_label            | source_kind   |   n_val_filtered |   target_iqr_Ha |   target_variance_Ha2 |   tmqm_val_gemnet_mae_Ha |   tmqm_val_schnet_mae_Ha |   qm9_mean_shift_gemnet_mae_Ha |   qm9_mean_shift_schnet_mae_Ha |
|:------------------------|:--------------|-----------------:|----------------:|----------------------:|-------------------------:|-------------------------:|-------------------------------:|-------------------------------:|
| Delta^xTB_ref,BoE       | linear        |             6542 |         0.05081 |               0.00245 |                   0.0064 |                   0.0093 |                        0.00615 |                        0.00703 |
| Delta^BoB_ref,BoE       | linear        |             6542 |         0.04113 |               0.0019  |                   0.0108 |                   0.0142 |                        0.01888 |                        0.04403 |
| Delta^MLP_BoB_ref,BoE   | MLP           |             6542 |         0.02597 |               0.00144 |                   0.0174 |                   0.0179 |                        0.03015 |                        0.02788 |
| Delta^PRE_ref,BoE       | linear        |             6542 |         0.02316 |               0.00064 |                   0.0123 |                   0.0125 |                        0.03316 |                        0.05327 |
| Delta^MLP_PRE_ref,BoE   | MLP           |             6542 |         0.02009 |               0.00108 |                   0.0181 |                   0.0182 |                        0.05981 |                        0.06015 |
| Delta^PRE+A_ref,BoE     | linear        |             6542 |         0.02388 |               0.00087 |                   0.0165 |                   0.0175 |                        0.03478 |                        0.04042 |
| Delta^MLP_PRE+A_ref,BoE | MLP           |             6542 |         0.01752 |               0.00084 |                   0.0168 |                   0.0167 |                        0.06997 |                        0.07036 |
