# Speech Enhancement Models

**Abstract**

Separating background noise or suppressing it from a given recording to extract clean speech is beneficial for both humans that have hearing aids, and machines, such as Automatic Speech Recognition systems. In recent years there have been various different experiments to develop a Speech Enhancement (SE) model or framework that would produce high-quality estimations based on classical algorithms or Deep Neural Network methods. Every research tries to evaluate and compare the new method with existing ones. Still, most often, a comparison is biased or insufficient because of the different setup of training data or different evaluation metrics used between each trial. The criteria used to corrupt the data with additive noise differ per research, like the evaluation strategy. This work aims to investigate what the most promising SE approaches are by replicating experiments and using exact training and testing criteria together with a same set of evaluation metrics to identify what produces better clean speech signal estimations.

Keywords: Speech Enhancement, Noise Suppression, Colour Noises, Clean Speech Estimation, Machine Learning Methods, Model Evaluation.

Content of repository:

| Folder | Description |
| ------ |------------ |
| FIGURES | Sample signals' figures of clean, noisy and denoised signals for each model and SNR (dB) level |
| MODELS | Source code for models and weights file of final molel|
| SIGNAL_SAMPLES | Sample signals for ech model and SNR (dB) level: clean, corrupted and denoised |

## Models investigated:
### DCCRN

DCCRN [1] is an autoencoder-based model with skip-connections between encoder and decoder layers employing LSTM transformations for a latent vector which is an output of the encoder before passing it to the decoder for reconstruction. This model works as an end-to-end framework but uses 1D convolutions to learn transform signal using Short-time Fourier transform into time-frequency representation and perform speech enhancement in this domain.

### DCUNET

DCUNet [2] is a complex ratio mask estimator, that employs U-Net principles. U-Net is a convolutional autoencoder with skip-connections. The whole architecture is based on complex-valued blocks that consist of complex convolution operations followed by a complex rectified linear unit (CReLU) and complex batch normalisation. This work uses a 20-layer version of DCUNet (see Figure 3) - 10 for encoder or down-sampling and 10 for decoder or up-sampling. As research paper analysing different variations of the model show that 20 layer model gives the best results on noise suppression.

### SEGAN

SEGAN [3] is the end-to-end speech enhancement framework working to map noisy to actual signal. It consists of two adversarial networks: generator (that produces actual signal) and discriminator (that decides if produced clean signal estimation is real). The discriminator network is used during the training phase to improve generator’s estimation quality. The generator network employs U-Net architecture with skip-connections (See Figure 4), making it similar to DAE as it has encoder and decoder modules. Differently from the DAE, encoder output is concatenated with random vector z, which is sampled from random Gaussian distribution with zero mean and unit variance promoting
model to use noise reconstructing actual signal.

### DTLN

DTLN [4] is an end-to-end LSTM-based network that works in both time-frequency and waveform
domains of the speech signal. This model estimates masks for both signal representations to achieve
clean and undisturbed speech. Similarly, as DCUNet and DCCRN the DTLN aims to minimise SDR
loss.

## References

[1] Y. Hu, Y. Liu, S. Lv, M. Xing, S. Zhang, Y. Fu, J. Wu, B. Zhang, L. Xie. DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement, CoRR, 2020.

[2] H. S. Choi, J. H. Kim, J. Huh, A. Kim, J. W. Ha, K. Lee. Phase-aware speech enhancement with deep complex u-net, International Conference on Learning Representations, 2018.

[3] S. Pascual, A. Bonafonte, J. Serrà. SEGAN: Speech enhancement generative adversarial network, CoRR, 2017.

[4] N. L. Westhausen, B. T. Meyer. Dual-signal transformation lstm network for real-time noise suppression, CoRR, 2020.