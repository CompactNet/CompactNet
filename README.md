# CompactNet
Purposive and Platform-Aware Optimization for Convolutional Neural Network

Convolutional Neural Network (CNN) based Deep
Learning (DL) has achieved great progress on many
real-life applications. Meanwhile, due to the complex
model structure against strict latency and memory
restriction, the implementation of CNN models on
the resource-limited platform is becoming more challenging.
This work proposes a solution, called CompactNet,
that automatically optimize a pre-trained
CNN model to be deployed on a specific resourcelimited
platform given a specific target of inference
speedup. Driven by a simulator of the target platform,
CompactNet purposively and progressively trims a
pre-trained network (by removing certain redundant
filters) until the target speedup is met and generate
an optimal platform-specific model while maintaining
the accuracy. We deploy our work on a smartphone
with two backend platforms of a mobile CPU
and a domain-specific accelerator. For image classification
on the Cifar-10 dataset, CompactNet achieves
up to a 1.8x kernel computation speedup with equal or
even higher accuracy compared with the state-of-the-art
slim CNN model made for the embedded platform
– MobileNetV2.

The whole paper will be published soon!


Instructions:

For the platform simulator:

Requirements:
1. TensorFlow (with Eager_Execution preferred);

2. Optional: Other platform SDKs (like HUAWEI HiAI) for Tensorflow
if you want to deploy it on other domain-specific accelerator

Running: 
1. Run python collector_eager.py (preferrd) to collect latency data
of each layer in MobileNetV2 with any number of input/output channels.
The data will be saved in /simulator/sim_data/

2. Run python evaluator.py to simulate the whole latency of a trimmed mode.

For the main searching loop:

Requirement:
1. TensorFlow with Keras API;

2. Latency data of the target platform collected by the simulator.

Running: 
1. Set the target_speedup, num_iters and the decay for the algorithm;

2. Run python searching.py to trim and generate the optimal platform-specific model.