# KineticRV
Datasets and data processing codes related to KiRV

# Ethics Statement

This multimodal gait recognition dataset, comprising visual and radar modality data, is curated with strict adherence to ethical standards to ensure responsible data governance, privacy protection, and transparent usage. The following ethical guidelines govern the collection, distribution, and utilization of this dataset:

All participants provided informed consent prior to data collection, with explicit awareness of potential research uses. Consent forms emphasize voluntary participation, the right to withdraw, and data anonymization protocols. Participants retain the right to request removal of their data from future distributions.

The dataset undergoes periodic audits to identify and mitigate demographic or environmental biases. Users are encouraged to report algorithmic biases or unintended discriminatory outcomes. The dataset prohibits applications that perpetuate harm, such as mass surveillance or profiling without legal justification. Both modalities comply with relevant international privacy regulations. Visual data is encrypted during transit/storage, with access logs maintained for accountability. Radar data links are version-controlled to prevent unauthorized modifications.

Radar modality data, including micro-Doppler spectrums, are publicly accessible. This data has undergone rigorous de-identification to remove personally identifiable information (PII), ensuring anonymity while preserving gait dynamics. The link is permanently hosted on a secure, institutionally maintained platform to guarantee long-term accessibility and integrity.

Vision modality data, including RGB video sequences, contains sensitive biometric information. To protect participant privacy, this data is not publicly available. Researchers seeking access to visual data must contact the corresponding author via email at 1272891929@qq.com to request approval. Requests will be evaluated based on ethical review board approval, research intent alignment, and adherence to privacy safeguards. Approved users must sign a data usage agreement prohibiting re-identification, commercial exploitation, or discriminatory applications.

This statement, alongside dataset documentation, is publicly available to foster transparency. Users must cite the dataset in publications and report derivative works. The authors assume no liability for misuse but commit to addressing ethical concerns raised by the community. For inquiries, ethical concerns, or access requests, please contact the corresponding author. This dataset is intended for advancing gait recognition research in healthcare, security, and human-computer interaction—not for unregulated or unethical applications.

# Data Processing File
Multi_readAdcData.m & parameter_setting.m --read radar data

MTI.m --Static clutter processing

time-frequency fft.m --range-frequency processing

stft.m --time-frequency processing

foreground_seg.py --Gait silhouette extraction based on traditional methods

u2net.py --Gait silhouette extraction based on the deep learning method

optical_flow_cal.py --optical flow sequence calculation method

# Datasets Link
--Supported by Baidu Netdisk

https://pan.baidu.com/s/1xITweCy8oFNjz6VIWJUHVg 

Extracted code：1202 
