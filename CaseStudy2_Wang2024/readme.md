Case study 2 in manuscript "Hierarchically structured data can undermine machine learning results in geoscience", by Scharf, Daggitt, Doucet and Kirkland.
This is 1) a reproduction and 2) an edit of code published by Wang et al. (2024). We look only at the model for the first step of zircon classification.
This work was completed using Python 3.13 in PyCharm 2024.3.4 (Community Edition).

Files of note:
- main_lgb_script.py: Wang et al. (2024) published a jupyter notebook. We have copied their code cell-by-cell and provide it in as a python script, commented with cell numbers. Minimal updates are added to ensure that this script runs with 1) the published datasets of Wang et al. (2024) and 2) the version of Python packages used at the time that this reproduction was done (listed in the requirements file).
- main_lgb_script_entity_splitting.py: An updates of main_lgb_script.py to execute a comparison of entity-splitting and observation-splitting methodologies.
- custom_functions.py: Additional functions NOT originally created by Wang et al. (2024). These faciliate plotting.
- requirements.txt: a list of Python packages used to complete this work.
- readme_Wang2024.txt: the original readme file of Wang et al. (2024), not to be confused with the readme file relevant to this project.

Folders of note:
- Outputs: A folder where outputs, primarily plots, are saved to.