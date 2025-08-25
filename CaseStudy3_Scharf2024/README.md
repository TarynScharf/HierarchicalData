Case study 3 in manuscript "Hierarchically structured data can undermine machine learning results in geoscience", by Scharf, Daggitt, Doucet and Kirkland.
This is 1) a reproduction and 2) an edit of code published by Scharf et al. (2024). We look only at the scenario 2 model for silica estimation.
This work was completed using Python 3.13 in PyCharm 2024.3.4 (Community Edition).

Files of note:
- Estimate-Silica.py: Original code of Scharf et al. (2024).
- Estimate-Silica_entity_split.py: An update of Estimate-Silica.py to execute a comparison of entity-splitting and observation-splitting methodologies.
- custom_functions.py: Additional functions NOT originally created by Scharf et al. (2024), which facilitate entity-splitting tests and data plotting.
- requirements.txt: a list of Python packages used to complete this work.
- readme_Scharf2024.md: the original readme file of Scharf et al. (2024), not to be confused with the readme file relevant to this project.

Folders of note:
- Outputs: A folder where outputs are saved to. 
  - Reproduction_Kfold_scenario_2_2AGGx2Resample_SHAPE_UTH_CL_24082025160859: The results of Estimate-Silica.py, reproducing kfold results of Scharf et al. (2024). 
  - Results_Kfold_scenario_2_2AGGx2Resample_SHAPE_UTH_CL_25082025121841: The results of Estimate-Silica_entity_split.py.