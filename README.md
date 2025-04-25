# FallWatch: A Fall Detection and Alert System

Name: Jiajie Hao  
Link to Github: https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system  
Link to Edge Impulse projects: https://studio.edgeimpulse.com/studio/653499  
Video presentation:  

## Introduction
With increasing global aging, falls are frequent among older people (Soffer et al., 2024). According to the World Health Organization (2007), about 28% to 35% of people aged 65 years and older experience at least one fall per year. This is highly susceptible to problems such as bone fractures and brain injuries in the elderly, which seriously affect their health and quality of life (Maruf et al., 2025). Especially if timely assistance is not available after a fall, the risk of mortality and complications increases significantly. Therefore, the development of a system that can monitor and warn of falls in real-time is important for the safety and health of the elderly.  

Based on this, this project proposes a fall detection and alert system (FallWatch). It collects motion data through a device fixed at the waist and uses deep learning algorithms to recognize both falling and normal standing postures. Once a fall is detected, the system will sound an alarm through a buzzer to alert nearby personnel to respond, providing more timely help and safety for the elderly. Initial tests show that the system can achieve almost 100% detection accuracy, validating its feasibility in real-time monitoring scenarios.  
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/fallwatch%20inside.jpg" alt="Project 1" style="width: 300px; height: auto; transform: rotate(-90deg); transform-origin: center;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/fallwatch.jpg" alt="Project 2" style="width: 300px; height: auto;">
</div>

## Research Question  
How to utilize wearable technology and deep learning algorithms to develop a reliable and accurate real-time fall detection system for the elderly, so that their falls can be detected in time and their safety and independence can be enhanced.  
## Application Overview   
In the Main Loop, the FallWatch system uses the LSM9DS1 IMU module, which is included in the Arduino Nano 33 BLE Sense Board to continuously collect data from the 9-axis inertial sensor (accelerometer accX/Y/Z, gyroscope gyrX/Y/Z, magnetometer magX/Y/Z) at 62.5Hz. The original data first enters the pre-processing module. High-frequency noise is suppressed through a low-pass filter (with a cut-off frequency of 5Hz), and the energy features in the frequency domain are extracted by combining the Fast Fourier Transform (FFT) to enhance the recognition of sudden violent movements in fall events. The processed multimodal signals are fed into the TensorFlow Lite interpreter. It solidifies the pre-trained lightweight neural network model in the form of a static array in the embedded memory to achieve the distinction between two types of actions: "falling" and "standing". Ultimately, the detection responder takes the probabilities output by the model and displays the pose judgment via the serial output. If a fall is detected, the indicator buzzer will give a continuous alarm to call for help. 
<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/application%20architecture.png"  alt="yellow" style="width: 500px; height: auto;">

## Data  
### Data Acquisition:  
This project collects data from two movement modes (standing and falling), with a total duration of 9m 42s, obtaining a total of 294 samples (50% of each type). Each sample lasts for 2 seconds and the sampling frequency is 62.5Hz. The ratio of training and test sets is 79%-21%.  
<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/dataset.png"  alt="yellow" style="width: 300px; height: auto;">  
<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/sampling%20frequency%20desicions.png"  alt="yellow" style="width: 500px; height: auto;"> 

To ensure data diversity, the data of this project is collected in three ways:
1) The experimental team members (the author and 6 classmates) simulated a variety of real fall postures (e.g., forward, backward, sideways) to collect 192 samples.
2) 21 sample data of fall/stand actions were obtained from the open-source dataset.
3) 81 samples were collected by directly throwing the Arduino board to simulate sudden impact.
   
In order to enhance data authenticity, the standing samples were additionally collected with disturbance data from daily activities (e.g., slight swaying and slow walking) in addition to the stationary state, covering the typical behavioral patterns of the elderly.
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/fall%20data.png" alt="Project 1" style="width: 300px; height: auto; transform: rotate(-90deg); transform-origin: center;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/stand%20data.png" alt="Project 2" style="width: 300px; height: auto;">
</div>

### Data pre-processing:   
For the motion characteristics at the initial stage of falling of elderly users (e.g., slow shift of the center of gravity), their low-frequency acceleration patterns (0-5Hz) are significantly different from normal standing posture. Therefore, the Low-pass Filter is used in this project and the cut-off frequency is set to 5Hz to retain the key low-frequency components of the elderly fall movement and improve the recognition accuracy of the model.  
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/fall%20data%20-%20after%20filter.png" alt="Project 1" style="width: 400px; height: auto;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/stand%20data%20-%20after%20filter.png" alt="Project 2" style="width: 400px; height: auto;">
</div>
<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/spectral%20features.png"  alt="yellow" style="width: 400px; height: auto;">  

## Model  
This project uses a concise designed, deep learning model based on the TensorFlow framework.   
### Model architecture:  
The model uses a Sequential structure and contains three fully connected layers (Dense layers):  

1)The input layer receives 54 feature data.  

2)The first dense layer has 16 neurons with ReLU activation function and L1 regularisation.  

3)The Dropout layer with a dropout rate of 0.3.  

4)The second dense layer has 8 neurons, also using the ReLU activation function and L1 regularisation.   

5)The output layer uses the Softmax activation function for outputting classification probabilities.   

<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/NN.png"  alt="yellow" style="width: 500px; height: auto;">  

Since I found during the experiment that the model continuously produced the phenomenon of overfitting, on the basis of appropriately reducing the learning rate and adding more data to the training dataset, I attempted to add a Dropout layer between the first and the second dense layer, enhancing the generalization ability of the model.  
### Training parameters:  
Number of training cycles: 40  

Learning Rate: 0.0003  

Batch Size: 32    

(This is a result of fewer training rounds and a better learning rate obtained through my experiments. For the specific process, please refer to the "Experiments" section)  

Finally, during the deployment phase, the model was converted to TensorFlow Lite Quantized (int8) format to optimize performance and reduce memory usage.
## Experiments  
### 1.Sampling frequency  
In the initial experiment, the data sampling frequency was set to 100Hz to the subtle signal changes of the falling action. However, this resulted in the number of raw features up to 1,800 per sample (9 × 2s × 100Hz = 1,800 points), and such a large amount of data may put considerable pressure on the memory and processing performance of the Arduino. In contrast, this quantity was reduced to 1125 (9 × 2s × 62.5Hz = 1125) when the sampling frequency was adjusted to 62.5Hz. In the comparison experiments, it was found that there was no significant difference in model accuracy between the two sampling frequencies. Therefore, considering the data volume and performance factors comprehensively, 62.5Hz was finally selected as the sampling frequency to optimize the system performance.  <br>

<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/sampling%20frequency.png"  alt="yellow" style="width: 500px; height: auto;">  <br>

From the test result of 62.5 Hz, the model showed an underfitting situation. Based on the expansion of the dataset, attempts were made to modify the Spectral Features to better capture the periodic and frequency information in the data and improve the discrimination accuracy of the model.  
### 2.Spectral Features  
<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/spectral%20features%20changes.png"  alt="yellow" style="width: 700px; height: auto;">  

Based on the above experimental process, and since the fall process of the elderly often shows a progressive imbalance pattern dominated by the low-frequency domain, it is concluded that the data features processed by the low-pass filter have more obvious differences and higher test accuracy. As a result, it was decided to adopt this feature mode. However, 100% accuracy may suggest overfitting, which is detrimental to the generalization of the model to new data. Therefore, the neural network settings will be subsequently adjusted to optimize the model performance.  
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/incorrect%20standing%20data1.png" alt="Project 1" style="width: 300px; height: auto;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/incorrect%20standing%20data2.png" alt="Project 2" style="width: 300px; height: auto;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/incorrect%20standing%20data3.png" alt="Project 2" style="width: 300px; height: auto;">
</div>  
Three standing actions with incorrect judgments  

### 3.Neuron Network Settings  
<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/NN%20settings1.png" alt="Project 2" style="width: 900px; height: auto;">
<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/NN%20settings2.png" alt="Project 2" style="width: 900px; height: auto;">  

Finally, the model complexity was reduced by decreasing the number of neurons, and the data set was also increased. As a result, both the training and test accuracy rates reached 100%. At this point, by observing the changes in the specific accuracy and loss values during the training process, it was seen that the model's accuracy tended to stabilize at epochs 38-40. Therefore, the 100% accuracy at this time showed no obvious signs of overfitting and could be adopted.  <br>

<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/training%20details(cycle45).png" alt="Project 2" style="width: 650px; height: auto;">
<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/training%20details(cycle40).png" alt="Project 2" style="width: 650px; height: auto;">  

## Results and Observations  
The model prediction accuracy is stable, with close to 100% accuracy during training. After deployment, the model was able to accurately identify fall and stand movement categories and output detailed prediction probabilities.  
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/stand%20still.png" alt="Project 1" style="width: 300px; height: auto;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/sway%20and%20stand.png" alt="Project 2" style="width: 300px; height: auto;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/fall.png" alt="Project 2" style="width: 300px; height: auto;">
</div>  
  <br>
Here are the three suggestions received during the Final pitch, along with the corresponding improvements and reflections:    
<br>  
<br>
<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/final%20pitch%20suggestions.png" alt="Project 2" style="width: 800px; height: auto;">  

### Effective work:   
The tuning of the neural network for the neural network has yielded significant results, as is evident from the following two graphs of the classification results. The left one shows the classification before tuning, while the right one reflects the results after optimizing the training period, learning rate, network structure, and number of neurons. It can be seen that the adjusted model is able to identify the features of the two classes of actions more accurately and the category boundaries are clearer.  
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/previous%20data%20explorer.png" alt="Project 1" style="width: 400px; height: auto;">
  <img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/data%20explorer%20after%20change%20NN.png" alt="Project 2" style="width: 400px; height: auto;">
</div>  

### Limitations:  
I observed that after deployment, the Arduino took much longer than the 2 seconds set by Edge Impulse to collect motion data, and actually took about 6 seconds. This may be because each time data is collected, the system needs to wait for all the sensors to be ready, and calling functions such as IMU.readAcceleration(), IMU.readGyroscope() and IMU.readMagneticField() will all introduce additional delays.  
<br>  

<img src="https://github.com/hml1688/CASA0018--Fall-detection-and-alert-system/blob/main/Images/time%20cost.png" alt="Project 2" style="width: 400px; height: auto;">

### Afterward, if I have enough time, I will:   
1. Collect more types (e.g., walking, sitting) and amounts of data to enrich the dataset. This will help to improve the generalization ability of the model, making it a more comprehensive model for elderly movement monitoring.

2. Equip the device with batteries to get rid of wires, and achieve the intelligent and portable wearable function.
## Bibliography  
1.Soffer, T., Raban, Y., Warshawski, S. & Barnoy, S. (2024) The impact of emerging technologies on healthcare needs of older people. Health Policy and Technology. 13 (5), 100935. doi:10.1016/j.hlpt.2024.100935.  

2.World Health Organization (2007) WHO global report on falls prevention in older age. Available at: https://extranet.who.int/agefriendlyworld/wp-content/uploads/2014/06/WHo-Global-report-on-falls-prevention-in-older-age.pdf (Accessed: 21 April 2025).  

3.Maruf, Md., Haque, Md.M., Hasan, Md.M., Farhan, M. & Islam, A. (2025) State-of-the-Art Review on Fall Prediction among Older Adults: Exploring Edge Devices as a Promising Approach for the Future. Measurement: Sensors. 101878. doi:10.1016/j.measen.2025.101878.  
 
4.Wikipedia (2025) Nyquist–Shannon sampling theorem. Available at: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem  (Accessed: 22 April 2025).  
## Declaration of Authorship  
I, Jiajie Hao, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.  
<br>

Jiajie Hao  

2025/4/21  

Word count of the main text: 1464  
