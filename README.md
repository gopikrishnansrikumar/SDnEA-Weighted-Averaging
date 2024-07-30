# SDnEA Weighted Averging (Algorithm Combinations for Time Series Anomaly Detection)

## Overview

SDnEA Weighted Averaging is a straightforward method for combining multiple anomaly detection algorithms. It leverages the accuracy of each detector to assign weights, providing a computationally efficient way to enhance detection performance. This README file provides a brief overview of the methodology, its implementation, and limitations.

## Methodology

Weighted averaging combines the outputs of multiple detectors by assigning weights based on their detection accuracy. Higher weights are given to detectors with better performance, ensuring that more reliable detectors have a greater influence on the final decision.

## Weight Calculation
The weights are computed using the Standard Deviation normalized Entropy of Accuracy (SDnEA) combining scheme. The formula for calculating the weight 
The weights are computed using the Standard Deviation normalized Entropy of Accuracy (SDnEA) combining scheme. The formula for calculating the weight for each detector is:

![img.png](img.png)

## Combine Detectors: 
Apply the weights to the outputs of the detectors and compute the weighted average.

## Running the Application: 
A web based application is build using streamlit and can be found in the root folder (application.py)
You can run the streamlit application using cmd terminal

## References
[1] A. B. Ashfaq, M. Javed, S. A. Khayam and H. Radha, "An Information-Theoretic Combining Method for Multi-Classifier Anomaly Detection Systems," 2010 IEEE International Conference on Communications, Cape Town, 2010, pp. 1-5, doi: 10.1109/ICC.2010.5501984. keywords: {Intrusion detection;Detectors;Entropy;Logic;Communications Society;Computer science;Paper technology;Computer networks;Peer to peer computing;USA Councils},
