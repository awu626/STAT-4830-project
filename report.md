# Problem Statement

## What are you optimizing? (Be specific)
Our group aims to optimize and fine-tune a GRPO model from Unsloth. Unsloth is a system that allows models to learn through GRPO (Group Relative Policy Optimization) with significantly less VRAM than before. This has made it possible turn standard models into reasoning models with custom rewards without using insane amounts of VRAM. We will be following Step 6 on Unsloth's website (https://unsloth.ai/blog/r1-reasoning) and optimizing a custom model to tackle standardized testing preparation.

## Why does this problem matter?
A GRPO-based model is able to independently be able to provide not only the answer to a standardized test question but also reasoning behind it. Knowing this, a model highly proficient in this will significantly increase the productivity of people studying for these tests, as it will be able to provide tailored explanations to any issue students encounter.

## How will you measure success?
We will measure success both numerically (by giving a practice test/practice bank) and recording scores and also qualitatively by analyzing the reasoning output and seeing if it makes sense.

## What are your constraints?
The main constraints involve compute limitations, as training a GRPO (Gradient-based Policy Optimization) model on standardized test reasoning tasks requires substantial resources. Memory constraints may arise when fine-tuning large models, especially with long-context reasoning tasks. Additionally, data constraints include the availability of high-quality standardized test datasets with detailed explanations, as reasoning-based models require strong grounding in logical and verbal reasoning.

## What data do you need?
We will need question and answer banks from standardized tests. Our thought is that SAT/ACT preparation will be the easiest to find.

## What could go wrong?
Several challenges could arise, including mode collapse, where the model overfits to certain types of reasoning patterns instead of learning general strategies. Sparse rewards in reinforcement learning can make optimization difficult, requiring careful reward shaping to avoid bad solutions. Lastly, scalability issues may arise if the GRPO model struggles with long-context reasoning in standardized test questions.


# Technical Approach

## Mathematical formulation (objective function, constraints)
Our objective function is a combination of the score the model can receive on a bank of standardized testing practice questions and also the perceived quality of its reasoning.

Specifically, we are looking at:

$$
\theta^* = \arg\max_{\theta} \mathbb{E}_{P_{\theta}} [ R(s, a) ]
$$

where R is the reward function and P is the policy.


## Algorithm/approach choice and justification
The primary approach is Gradient-based Policy Optimization (GRPO), a policy gradient method that improves the model’s reasoning ability via reinforcement learning (RL). Justification for GRPO:

Handles long-horizon reasoning → Unlike supervised fine-tuning, GRPO can optimize multi-step reasoning paths.
Adaptive improvement → Uses reward-based updates to reinforce better reasoning chains.
Avoids overconfidence → Unlike standard transformers, GRPO gradually improves response distribution through sampled trials.


## Implementation strategy
We will use standardized test datasets (MMLU, ARC, SAT/GRE datasets), and tokenize these reasoning-based questions. We'll start with a baseline model like Unsloth LLaMA-3 8B (or maybe something smaller, depending on training time). Then, we'll implement a supervised fine-tuning baseline with GRPO. It will have states (contexts) and actions (responses), and we will train and evaluate this. Finally, we will tune it on more complex reasoning tasks.


## Validation methods
We will look at the score the tuned model is able to achieve on different test banks. We will also look into Chain-of-Thought (CoT) metrics to evaluate how strong the reasoning is. Finally, we will use limited human validation to assess whether things make sense or not.


## Resource requirements and constraints
Training time and capacity will likely be a challenge for us. Unsloth specifies that a bare minimum of 7 GB VRAM is required. We have at least two local GPUs within the group that have more than this, and also the Colab-provided Tesla T4 GPU. However, we noticed while running initial experimentation that the Tesla T4 trained quite slowly. We may consider moving to a smaller model (2.5B parameters would be a decent option) if training is too lengthy.


# Initial Results

## Evidence your implementation works
We started off by running the stock notebook provided and experimenting with it to get some training results.

![image](https://github.com/user-attachments/assets/0a5f29d5-4c74-48ad-b077-e78b23f7a750)


## Basic performance metrics
The base notebook trained on the T4 GPU at a rate of about 1 step per minute, and it was able to collect some reward. Due to free-tier Colab limits, we were unable to progress further but this will be rectified in the future with local training or Colab Pro. 

## Test case results
We observed that the rewards started increasing very slowly at first, but eventually picked up more steam. Average-of-5 rewards for steps seem to be going up steadily as the base model trains.

## Current limitations
Our biggest issue right now is compute speed as the training is very slow on T4, and also the amount of compute units provided in the free tier is very limiting.

## Resource usage measurements
With this current model (8B parameters), we observed a decent chunk of the T4's 16GB VRAM being used up to train.

## Unexpected challenges
Most of the challenges we have faced so far are expected. Training is slower than expected but there are ways to rectify that in the future.

# Next Steps

## Immediate improvements needed
We will try to get the models to start running locally to improve training times. The local GPUs we have hold 12 GB VRAM, but are nearly 3x faster on benchmarks. We believe they will be able to cut down on processing time. After that, we can look into actually fine-tuning the model.

## Technical challenges to address
We need to nail down a balance between a numerical metric and a human-based one and weigh those properly to judge reasoning performance. Since this model will be used to directly guide humans, it needs to output satisfactory if not great results that humans can understand.

## Questions you need help with
How valid is it to have human-based qualitative performance assessments along with numerical ones?

## Alternative approaches to try
There are many base models that can be experimented with, with differing logics and parameter sizes. We could look into some of those if we find that training is too slow or the bigger model is overkill. Additionally, we have considered looking into the actual training process and optimizing that instead of results.

## What you've learned so far
We have learned a lot about GRPO and reasoning models as this is our groups' first time looking into them in depth.
