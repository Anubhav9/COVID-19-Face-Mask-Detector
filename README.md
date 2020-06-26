# COVID-19-Face-Mask-Detector
Working in Real Time, this project detects whether a person has worn a face mask or not.

COVID-19 has been one of the most difficult phase for all of us. We are quite sure of the fact that things will not be the same in the Pre-COVID and Post-COVID
world. We all have been saying, "this is the new normal" and so continuing on this fact,we know for sure that wearing face mask in public will be compulsary
for time being. 

But it is not always possible for authorities to mannually keep a track on all of us. I have tried to build a Face Mask Detector which captures your face in
real time and predicts whether you are wearing a face mask or not.

The following things have been used:

i) OpenCV for real time detection

ii) Harcascadde Frontal Face Default XML File for getting the region of interest.

iii) A CNN has been used to train the model. After training the model, it has been stored as detect.h5 file.

The dataset has been taken from Pranja's Github Repo (https://github.com/prajnasb/observations/tree/master/experiements/data), so due credits to her as well.

To run this project, clone this repository and run the trial.py file using command - python3 trial.py.

Model Weights have already been saved, so you don't need to train your model again.

