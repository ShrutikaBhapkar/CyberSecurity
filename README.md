# CyberSecurity
Real Cybersecurity Data for Anomaly Detection Research


The BETH dataset currently represents 8,004,918 events collected over 23 honeypots, running for about five noncontiguous hours on a major cloud provider. For benchmarking and discussion, we selected the initial subset of the process logs. This subset was further divided into training, validation, and testing sets with a rough 60/20/20 split based on host, quantity of logs generated, and the activity logged—only the test set includes an attack

The dataset is composed of two sensor logs: kernel-level process calls and network traffic. The initial benchmark subset only includes process logs. Each process call consists of 14 raw features and 2 hand-crafted labels.


