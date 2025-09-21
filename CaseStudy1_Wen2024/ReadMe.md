Case study 1 in manuscript "Hierarchically structured data can undermine machine learning results in geoscience", by Scharf, Daggitt, Doucet and Kirkland.

This is 1) a reproduction and 2) an edit of code published by Wen et al. (2024). We look only at the PCR model.
This work was completed using Python 3.13 in PyCharm 2024.3.4 (Community Edition).


Files of note:
- main_PCR_reproduction.py: A copy of the code published by Wen et al. (2024), containing minor updates to ensure that this script runs with 1) the published datasets of Wen et al. (2024) and 2) the version of Python packages used at the time that this reproduction was done (listed in the requirements file).
- main_PCR_entity_splitting.py: An update of main_PCR_reproduction.py to execute a comparison of entity-splitting and observation-splitting methodologies.
- custom_function.py: Additional functions NOT originally created by Wen et al. (2024). These facilitate plotting.
- requirements.txt: a list of Python packages used to complete this work.
- ReadMe_Wen2024.txt: the original readme file of Wen et al. (2024), not to be confused with the readme file relevant to this project.

Folders of note:
- Outputs: A folder where outputs, primarily plots, are saved to.
