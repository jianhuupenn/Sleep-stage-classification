# Sleep stage classification based on Recurrent neural networks using wrist-worn device data
### Jian Hu, Haochang Shou*

![Workflow](docs/asserts/images/workflow.jpg)
Disturbances in sleep and anomaly movement are known to be closely related with various clinical endpoints. Traditionally, sleep detection and evaluations were mostly assessed via self-reported sleep diary or using polysomnography (PSG) in sleep clinics. Sleep diary, as the most commonly used approach in logging sleep-related information such as time of bed, time of rise and number of awakenings during sleep, relies entirely upon subjects’ recall. Hence the data are subject to recall bias and could suffer from missing information. PSG, on the other hand, are often used as the gold standard for objectively diagnosing sleep disorders. PSG combines detailed recordings of heart rate, blood oxygen, breathing and eye/leg movements to detect onset of sleep and classify sleep stages. However, since PSGs are mostly  limited within the sleep clinic and could be costly, data are rarely available for more than one day. Wearable sensor devices such as accelerometers have been reported to be highly correlated with PSG in monitoring sleep and wakefulness1.  With the increasing use of accelerometry tracking in many large-scale population studies such as the UK Biobank and National Health and Nutrition Examination Study (NHANES), multiple days of 24hr accelerometry data are often available from a large number of study subject. Hence it becomes a great interest for researchers to develop automatic algorithms that can reliably extract important features of sleep and circadian rhyme based on accelerometry tracking that are collected from subjects’ free-living conditions. 

Data obtained from accelerometers are often processed as time series of accelerations in three dimensions into frequency as high as 5s epoch grids. Additional data such as light exposure and ambient temperature are also available depending on the device. Early methods mainly focused on detecting the binary sleep/awake cycle and often relied upon arbitrary cut points using one or a few accelerometry features.  Recent developments started exploring machine learning methods for such task including Hidden Markov Models (HMM) and neural networks. However, few papers have explored the ability of actigraphy in classifying beyond the binary sleep/awake status, and further differentiating rapid eye movement (REM) versus non-REM. Additionally, we aimed to utilize a full set of features that were collected from accelerometers such as 3-axis accelerations, angles and variations and examined the performance of the most advanced machine learning methods over a relatively limited sample size. In particular, we experimented several sequential neural networks that are tailored for time series data. Recurrent Neural Networks are a class of sequential neural networks that can process a sequence of inputs and retain its state while processing the next sequence of inputs. Thus, they are able to remember the previous information when predicting the current state. Taking the advantage of time series information in such an efficient way, sequential neural networks are expected to have better performance compared to the non-sequential models.  In addition to the standard RNN, we also implemented the Long Short Term Memory Network (LSTM) for this task. LSTM is a special variant of recurrent neural network which utilizes a 'memory cell' that can maintain information in memory for long periods of time. Compared to standard RNN, LSTM is able to learn longer-term dependencies and is proved to show superior results in many publications. Several recent work has shown success in predicting sleep stages using ‘deep temporal modeling’ such as LSTM based on a comprehensive set of PSG signals  

Despite of the existing studies for sleep stage prediction using sensor data and advanced machine learning methods, most of them require input from multiple devices with several domains of biological signals and large number of participants in the training data, which poses challenges for collecting device data from in-home use. Our paper aims to asses whether machine learning (ML) models can achieve a good performance in sleep stages classification with features extracted solely from wrist-worn accelerometers and limited training sizes. 



For thorough details, see the preprint: [Bioxiv]()
<br>

## Contributing
Souce code: [Github](https://github.com/jianhuupenn/Sleep-stage-classification)  
Author email: jianhu@pennmedicine.upenn.edu
<br>
We are continuing adding new features. Bug reports or feature requests are welcome.
<br>

## Debugging

## Reference

Please consider citing the following reference:
