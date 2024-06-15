This is a repository for python-based code used to run an ECG variational autoencoder + dense network classifier for prediction of ARVC diagnosis.

Running the main file will excute the following actions:

1) Load a sample ECG ".xml" file and view the results in standard 12-lead ECG format
2) Use a Pan-Tompkins based algorithm to identify QRS and extract median beat data for each of the 12 leads, then view median beat data in 12-lead format
3) Load the pre-trained ECG variational autoencoder (VAE), and demonstrate median beat reconstruction of the sample ECG. a) note - this relies on tensorflow python deep-learning library and requires access to GPU. b) Input to the ECG VAE is 12 lead, 250Hz, 1.2s median-beat voltage wave-form data.
4) Print the latent variable (n=24) encoding of the sample ECG
5) Load dense-network classifier model for ARVC diagnosis
6_ Generate predictions for presence/absence of ARVC diagnosis; first - the raw continuous prediction is provided, second - the scaled prediction which can be incorporated into existing TFC is provided, third - dichotomous predictions at a variety of predefined specificities are provided.
