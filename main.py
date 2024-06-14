import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import ecg_reader
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


if __name__ == "__main__":

    # note - this code does require GPU access to run

    print('Use the ECG VAE for Diagnosis of ARVC')

    # Load a sample ECG saved in XML format
    # --- note - based on specific XML file formatting, ECG reader xml extraction may need to be locally adjusted
    # view 12 lead ECG with QRS identification
    # view the median beat identification algorithm

    samp_num = 1 # A true ARVC diagnosis
    samp_num = 2 # ARVC is absent

    ecg = ecg_reader.extract_ecg_xml_muse(f'Sample_ECG\\Sample_ECG_{samp_num}')
    ecg.plot_ecg()
    plt.suptitle(f'12 Lead View - Sample {samp_num}')
    ecg.plot_median_beat(plot_all_beats=0)
    plt.suptitle(f'Median Beat Identification for Sample {samp_num}')
    heart_rate = np.round(ecg.freq/np.mean(np.diff(ecg.qrs_peaks))*60)

    # Load the VAE
    # visualize reconstruction of the sample ECG
    # print the latent variable encoding of the ECG

    model = tf.keras.models.load_model('VAE_ECG')

    ecg_reconstruction = ecg_reader.empty_ecg()
    enc_mean, enc_logvar, encoding = model.encoder(ecg.median_beat[tf.newaxis, ...])
    ecg_reconstruction.median_beat = model.decoder(encoding)[0,:,:]

    fig, gs = ecg.plot_median_beat(plot_all_beats=0,color='black',new_fig=1)
    ecg_reconstruction.plot_median_beat(color='r',new_fig=0, old_fig=[fig, gs], alpha=0.7)
    plt.legend(['Original', 'Reconstruction'], loc='lower left')
    plt.suptitle(f'Reconstruction of Sample {samp_num}')

    print(f'\nLatent variables for Sample {samp_num}:')
    print(np.array(encoding)[0])

    # Load classifiers for ARVC Diagnosis that uses VAE-Encoder Latent Variable Encodings as Input

    # print the model predicted presence/absence of ARVC Diagnosis
    # print the overall recommendation for CMR based on presence of ANY high-risk feature

    sensitivities = [80, 90, 95, 98]
    cutoffs = [0.52, 0.37, 0.26, 0.15] # cutoffs based on 80%, 90%, 95%, and 98% sensitivity for ARVC Diagnosis

    print()

    classifier = tf.keras.models.load_model('Classifier_ARVC_Dx')

    continuous_prediction = classifier(encoding)[0][0].numpy()
    print(f'Automated ECG Assessment for ARVC Diagnosis: Sample {samp_num}')
    print(f'Model Prediction: {continuous_prediction:.3f}')
    print(f'Equivalent ECG-Derived TFC score: {continuous_prediction*4:.2f}')

    for i, cutoff in enumerate(cutoffs):

        print_line = f'At Sensitivity of {sensitivities[i]}% - ARVC is '

        if continuous_prediction > cutoff:
            dichotomous_prediction = 'Present'
        else:
            dichotomous_prediction = 'Absent'
        print(print_line + dichotomous_prediction)


    plt.show()
