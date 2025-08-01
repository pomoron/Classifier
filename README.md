# A Multi-defect Classifier using Images

This is a repo for using images to detect road defects on and off the pavement. The authors will update as soon as there are new developments in the project, notably in improving the detection, better georeferencing (referencing by object instead of by image) and be more flexible in accepting road images at different perspectives. Stay tuned with the latest developments 🚀

# Methods
* This setup detects road defects on and off the pavement.
* On the pavement, this repo aims to detect the following with the defect location 🛣️:
  * cracks (transverse, longitudinal, alligator)
  * potholes
  * patches
  * worn lane marking
  * (that's why orthorectified images and their locations are needed)
* Off the pavement, this repo aims to detect the following with locations of where the image was taken 🛑🌳:
  * overgrown vegetation
  * rusty barriers
  * signs
  * obstructions on the roads
* The repo currently runs with the bootstrapped method, with more upcoming improvements planned 🔜

# Installation
* ```git clone``` this repo to your local computer
* ```pip install``` with the requirements in ```requirements.txt```

# Operation Instructions
* This repo takes inputs of *orthorectified pavement images* and *front views of panoramic images* for detection
  * Pavement: it is just how the detection weights are being trained at the moment (utilising synthetic data created in Lam 2025). Currently pavement defects are detected separately from the off-pavement defects. Users can replace the weights with ones trained with perspective images; they should equally work with the aggregator.
  * Front views: dash camera images should work just as well. The detection process does not involve using custom fine-tuned models.
* Specify the input directories in ```config/paths.py```
  * Image directories
  * Georef files directories (for the aggregator)
  * Weights (download from releases and add the directories)
  * VLM projects (currently connects to a project in Google Cloud Platform, can connect to the Gemini API with an API key instead - free of charge in exchange of contributing your data)
* The modules to be run are written in ```run.py``` of the parent directory
  * The detection routines are written as a function imported from different sub-directories under ```modules```
  * Detection of each kind of defect runs independently in series.
  * Detection of each kind of defect will return with a datafile stored in ```output/intermediate``` as the default location (changeable at ```config/paths.py```)
* The final command runs the aggregator, which sums up defects detected in all detectors into a datafile.
  * Quote the geolocations from the georef files to find the X,Y,Z of each defect

# Planned developments
* (Detection) Potholes - fuse predictions based on image patterns and depth estimations
* (Detection) Overgrown vegetation - use YOLO weights to predict vegetation and pavement to save detection time from Florence-2 and SAM
* (Detection) Obstructions - multi-agent approach to reduce hallucination from VLM
* (Localisation) Attribute non-pavement defects to an asset instead of to individual images

# References
* Creating synthetic pavement defects (for the trained weight)
```bibtex
@article{lam_chen_de silva_brilakis_2025, title={Integrating Multi-Source Visual Synthetic Data for Multi Road Defect Detection}, url={https://www.repository.cam.ac.uk/handle/1810/383988}, DOI={10.17863/CAM.118150}, publisher={Apollo - University of Cambridge Repository}, author={Lam, Percy and Chen, Weiwei and De Silva, Lavindra and Brilakis, Ioannis}, year={2025} }

