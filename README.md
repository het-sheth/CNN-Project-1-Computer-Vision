CNN Forward Pass Implementation

The project focuses on building and evaluating the forward pass of a pre-trained, 18-layer Convolutional Neural Network (CNN) for object recognition on the CIFAR-10 dataset.

Project Overview:
The primary goal of this project was to implement the core computational layers of a CNN from scratch using fundamental MATLAB operations. The network takes a 32x32 color image as input and outputs a probability distribution over 10 object classes. This project covers the inference stage of a deep learning pipeline, where a pre-trained model is used to make predictions.

The key objectives achieved include:

    Implementing modular functions for each type of CNN layer.

    Assembling these functions into a full 18-layer pipeline.

    Evaluating the network's performance on the 10,000-image CIFAR-10 test set.

    Exploring the network's behavior with custom, out-of-class images.

Key Results

After running the full performance evaluation, our implementation achieved:

    Overall Top-1 Accuracy: 43.71%

        This is over 4 times better than random guessing (10%), demonstrating that the network successfully learned meaningful visual features.

    Top-3 Accuracy: Over 70%

        This shows that even when the top guess is incorrect, the true class is often within the network's top three predictions.

The network made logical errors based on visual similarity (e.g., confusing 'cats' and 'dogs'), but also revealed interesting behaviors, such as confusing 'airplanes' and 'ships', likely due to similar blue backgrounds in the training data.
How to Run the Project

All scripts are designed to be run from the MATLAB command window, provided the required .mat data files are in the same directory.
Main Demonstration

To see a full demonstration of the CNN processing a single test image and visualizing key results, run the main demo script:

>> run_project_demo

This script will produce several figures showing the original image, intermediate feature maps, and the final classification result.
Performance Evaluation

To replicate our final performance results, run the evaluation script. Note: This will take several minutes to process all 10,000 images.

>> run_performance_evaluation

This will output the final accuracy and the 10x10 confusion matrix to the command window.
Project Structure

The project is organized into a set of modular functions and high-level scripts:
Core Layer Functions (apply_*.m)

These are the fundamental building blocks of the network. Each function is self-contained and heavily commented.

    apply_imnormalize.m

    apply_convolve.m

    apply_relu.m

    apply_maxpool.m

    apply_fullconnect.m

    apply_softmax.m

Main Scripts

    run_cnn_forward_pass.m: The central "engine" function that executes the full 18-layer pipeline for a single image.

    run_project_demo.m: The main demo routine for showcasing the project's functionality.

    run_performance_evaluation.m: The script used to generate the final accuracy and confusion matrix.

    run_custom_image_test.m: A script for exploratory testing on custom images.

    run_top_k_evaluation.m: The script for the extra credit Top-K analysis.

Data Files (Not included in this repository)

The scripts require the following .mat files to be present in the root directory:

    cifar10testdata.mat

    CNNparameters.mat

    debuggingTest.mat
