# Problem Statement

## What are you optimizing? (Be specific)
Our group aims to optimize and fine-tune a GRPO model from Unsloth. Unsloth is a system that allows models to learn through GRPO (Group Relative Policy Optimization) with significantly less VRAM than before. This has made it possible turn standard models into reasoning models with custom rewards without using insane amounts of VRAM. 
We will be following Step 6 on Unsloth's website (https://unsloth.ai/blog/r1-reasoning) and optimizing a custom model to tackle standardized testing preparation. More specifically, we want to have a model that can help people study for the SAT and explain all of the correct steps they need to take to do so.

## Why does this problem matter?
A GRPO-based model is able to independently be able to provide not only the answer to a standardized test question but also reasoning behind it. A highly proficient model could greatly improve study efficiency by offering tailored explanations for different problems. 
Similar reinforcement learning-based methods, like DeepSeek-R1 (DeepSeek-AI, 2025) and Math-Shepherd (Wang et al., 2024), have shown success in enhancing reasoning skills in language models. We think this problem matters because standardized testing is something that a lot of students struggle with. It can be hard to study for, especially
if you're not a good test taker, and it can feel overwhelming. A model like this could be integrated into AI assistants to better explain SAT preparation in a way that is more concise than the sometimes long-winded answers ChatGPT gives.

## How will you measure success?
We will measure success both numerically (by giving a practice test/practice bank) and recording scores and also qualitatively by analyzing the reasoning output and seeing if it makes sense.

## What are your constraints?
The main constraints involve compute limitations, as training a GRPO (Gradient-based Policy Optimization) model on standardized test reasoning tasks requires substantial resources. Memory constraints may arise when fine-tuning large models, especially with long-context reasoning tasks. Additionally, data constraints include the availability of high-quality standardized test datasets with detailed explanations, as reasoning-based models require strong grounding in logical and verbal reasoning. 

## What data do you need?
We have question and answer banks totalling multiple thousands of questions. They are formatted as question and answer in one data point.

## What could go wrong?
Several challenges could arise, including mode collapse, where the model overfits to certain types of reasoning patterns instead of learning general strategies. Sparse rewards in reinforcement learning can make optimization difficult, requiring careful reward shaping (Setlur et al., 2024) to avoid bad solutions. 
We are currently worried about finding the correct reward function to get it to actually learn properly.


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

- Handles long-horizon reasoning → Unlike supervised fine-tuning, GRPO can optimize multi-step reasoning paths.
- Adaptive improvement → Uses reward-based updates to reinforce better reasoning chains, following methods explored in DeepSeekMath (Shao et al., 2024).
- Avoids overconfidence → Unlike standard transformers, GRPO gradually improves response distribution through sampled trials.

## Implementation strategy
We will use standardized test datasets (MMLU, ARC, SAT/GRE datasets), and tokenize these reasoning-based questions. We are currently using the A100 GPU to train on Llama 3 (8B), which delivers reasonable speeds.


## Validation methods
We will look at the score the tuned model is able to achieve on We will evaluate model performance using:
- **Quantitative metrics:** Standardized test scores on benchmark datasets.
- **Qualitative assessment:** Chain-of-Thought (CoT) evaluations and human validation to ensure logical reasoning quality.


## Resource requirements and constraints

We don't anticipate any further resource constraints; most of the problems will be technical.


# Initial Results

## Evidence your implementation works
We started off by running the stock notebook provided and experimenting with it to get some training results.

![image](https://github.com/user-attachments/assets/0a5f29d5-4c74-48ad-b077-e78b23f7a750)

We also now have a system in place to transfer formulas in nested format:

multiply(subtract(add(2, 4), 2), 4)

into a format that is readable by the Python library SymPy (symbolic math). This will allow us to use the LLM to get the formula from the question rather than go straight to the answer.

We anticipate this method to be easier and also give us a place to get reasoning, as the formula is indirectly representing the reasoning.


## Basic performance metrics
The A100 is many times (at least 4-5) faster than the T4 which we used previously, which has helped us massively.

## Test case results
We observed that the rewards started increasing very slowly at first, but eventually picked up more steam. Average-of-5 rewards for steps seem to be going up steadily as the base model trains.

## Current limitations
The main limitation right now is the reward function. We have still been unable to find one that can help us get the right answer and the model hallucinates a lot.

## Resource usage measurements
Training eats up a significant chunk of the A100's VRAM, but it is not maxed out meaning there is overhead somewhere else. Training will not go faster than this.

## Unexpected challenges
We are currently figuring how exactly we want to get the LLM to learn to transform the problem into the formula.

# Next Steps

## Immediate improvements needed
1. Improve training to get the formula
2. Expand Literature Review


## Technical challenges to address
We need to nail down a balance between a numerical metric and a human-based one and weigh those properly to judge reasoning performance. Since this model will be used to directly guide humans, it needs to output satisfactory if not great results that humans can understand.

We also need to attempt at balancing training time with reasoning quality, to find the right trade-off between training speed and performance.


## Questions you need help with
How valid is it to have human-based qualitative performance assessments along with numerical ones?

## Alternative approaches to try
There are many base models that can be experimented with, with differing logics and parameter sizes. We could look into some of those if we find that training is too slow or the bigger model is overkill. Additionally, we have considered looking into the actual training process and optimizing that instead of results.

## What you've learned so far
We have learned a lot about GRPO and reasoning models as this is our groups' first time looking into them in depth.

# Literature Review

Our literature review examined several sources that contributed to our understanding of GRPO models and reinforcement learning techniques applied to reasoning tasks.

- **DeepSeek-R1 (DeepSeek-AI, 2025)** demonstrated how reinforcement learning with GRPO can significantly improve the reasoning capabilities of LLMs. The study found that reward-based fine-tuning improved the model’s ability to solve math, coding, and standardized test questions without requiring extensive human annotation. This validates our choice of GRPO as a reinforcement learning framework for our project.

- **DeepSeekMath (Shao et al., 2024)** introduced GRPO as a variation of PPO, demonstrating how it can be used to train math reasoning models efficiently. The key takeaway from this paper was that GRPO avoids the need for a critic network by using group-based baseline rewards, which could significantly reduce memory and computational costs in our project.

- **Math-Shepherd (Wang et al., 2024)** proposed a reward model that improved step-by-step verification of solutions in math problem-solving tasks. This highlighted the importance of reward shaping in RL-based reasoning models, which we plan to incorporate into our own fine-tuning process.

- **Setlur et al. (2024)** explored reinforcement learning on incorrect synthetic data, showing that fine-tuning on negative samples dramatically increased training efficiency. This suggests that our training data can be augmented with incorrect answers to reinforce model learning, making training more sample-efficient.
