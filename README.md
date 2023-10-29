[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/BnSnqDzS)
# Coding assignment: Fine-grained image classification of Korean food

## Introduction

We will build a neural network to classify kind of Kimchies. You need to build a deep neural network to classify the 11 detailed classes of Korean Kimchie; ['갓김치', '깍두기', '나박김치', '무생채', '배추김치', '백김치', '부추김치', '열무김치', '오이소박이', '총각김치', '파김치'].

**Note**: Use your favorite deep learning library.

---
## Deadline
November 15, 2023 2:00PM KST (*No delay is permitted.*)

### Submission checklist
* Push your code to [our github classroom page's 'Coding assignment' section](https://classroom.github.com/a/BnSnqDzS)
* Submit your report to [Gradescope 'Coding assignment' section](https://www.gradescope.com/courses/624027)

---
## Preparation

### Download the dataset

1. Register you in [NIA's dataset lake](https://aihub.or.kr/join/mberSe.do?currMenu=108&topMenu=108)
1. Go to [dataset page in NIA for this project (Korean food dataset)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=79) to download the dataset. Make sure to download all data
    - Note that the entire dataset is 15.73GB but we are going to use a portion of it.

### Prepare the dataset

1. Copy the dataset to the code directory
1. Run the following command in a shell (`cmd` or `console` in Windows -- We only test this in Unix like OS such as Mac OS)
```
$ Python prepare_dataset.py
```

---
## What to submit
**Push your code to your github classroom** 

- All of the python files you have implemented (under "Files you'll edit"). 
  - **Caution:** DO NOT UPLOAD THE DATASET

**Upload your report to Gradescope**
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).
  - If you do not want to use LaTeX, use any other wordprocessor and render it to PDF.



---
## Classification with neural network classifier - deep multi layer perceptron (MLP) (30%)

Perform image classification on the given dataset using a deep multi layer perceptron.

`REPORT 1-1`: Describe model details (hyperparameters) your have used (e.g., initial learning rate, weight decay and etc.)

`REPORT 1-2`: Why did you choose to use these hyperparameters? (e.g., using cross validation)

`REPORT 1-3`: Report both the training and testing accuracy in a plot (x: epoch, y: accuracy). 

`REPORT 1-4`: Plot class-wise test accuracy.

`REPORT 1-5`: What is the possible reasons for the bad performance in some classes

`REPORT 1-6`: Discuss any ideas to improve the accuracy

`REPORT 1-7`: Apply your idea to improve the accuracy

`REPORT 1-8`: Plot class-wise test accuracy and the average test accuracy.

---
## Classification with deep convolutional neural network (CNN) (70%)

Perform image classification on the given dataset using a deep convolutional neural network.
Using your choice of deep neural network frameworks (e.g., PyTorch, Tensorflow), implement convolutional neural network (CNN) without extra feature transform.

`REPORT 2-1`: Describe model details (hyperparameters) your have used (e.g., initial learning rate, weight decay and etc.)

`REPORT 2-2`: Why did you choose to use these hyperparameters? (e.g., using cross validation)

`REPORT 2-3`: Report both the training and testing accuracy in a plot (x: epoch, y: accuracy). 

`REPORT 2-4`: Plot class-wise test accuracy.

`REPORT 2-5`: What is the possible reasons for the bad performance in some classes

`REPORT 2-6`: Discuss any ideas to improve the accuracy

`REPORT 2-7`: Apply your idea to improve the accuracy

`REPORT 2-8`: Plot class-wise test accuracy and the average test accuracy.

`REPORT 2-9`: Discuss the difference to the MLP based model.

---
### Note
**Academic dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting help:** You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, class time, and Piazza are there for your support; please use them. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.
