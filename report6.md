# Problem Statement

## What are you optimizing? (Be specific)
Our group aims to optimize a Transformer model on a dataset of SAT problems to output a formula, which can then be parsed through Regex and passed through sympy to get a correct answer.

## Why does this problem matter?
A Transformer-based model is able to independently be able to provide not only the formula to a standardized test question but also reasoning behind extracting the formula in that manner. A highly proficient model could greatly improve study efficiency by offering tailored explanations for different problems. 
Other methods, like DeepSeek-R1 (DeepSeek-AI, 2025) and Math-Shepherd (Wang et al., 2024), have shown success in enhancing reasoning skills in language models. We think this problem matters because standardized testing is something that a lot of students struggle with. It can be hard to study for, especially
if you're not a good test taker, and it can feel overwhelming. A model like this could be integrated into AI assistants to better explain SAT preparation in a way that is more concise than the sometimes long-winded answers ChatGPT gives.

## How will you measure success?
We will measure success both numerically, by seeing if our model outputs the correct answers.

## What are your constraints?
The main constraint has to do with the data. While it is made up of 30,000 examples, it is not the cleanest or best put together data and it has some grammar mistakes.

## What data do you need?
We have question and answer banks totalling multiple thousands of questions. We also have a column for a functional formula to solve each question.

## What could go wrong?
We are concerned about exactness, as our SymPy based architecture for solving the problem needs exact Regex to work properly. If the formatting is off, it will lead to an incorrect answer even if the formatting is 99% there.

# Technical Approach

## Mathematical formulation (objective function, constraints)
Our objective functions are twofold. Firstly, the percentage of answers it gets completely correct. Second, how close the model outputs are to the ground truth, even if t hey are not exactly correct.

Specifically, we are looking at:

$$
\M^* = \arg\max_{\M} M(E) = E^*
$$

where M(E) is the output of the model on problem E and E* is the ground truth as well as

$$
\M^* = \arg\max_{\M} SIM(M(E), E^*)
$$

where M(E) and E* are the same as above, and the SIM() function measures the similarity ratio between the outputs.

## Algorithm/approach choice and justification
The initial approach we took was Gradient-based Policy Optimization (GRPO), a policy gradient method that improves the model’s reasoning ability via reinforcement learning (RL). Justification for GRPO:

- Handles long-horizon reasoning → Unlike supervised fine-tuning, GRPO can optimize multi-step reasoning paths.
- Adaptive improvement → Uses reward-based updates to reinforce better reasoning chains, following methods explored in DeepSeekMath (Shao et al., 2024).
- Avoids overconfidence → Unlike standard transformers, GRPO gradually improves response distribution through sampled trials.

However, we have started experimenting with different types of transfomer architecture. Transformers are great at extracting data from text and establishing long-term text dependencies, a skill required for solving math problems.

## Implementation strategy
We will use standardized test datasets (MMLU, ARC, SAT/GRE datasets), and tokenize these reasoning-based questions. We are currently using the A100 GPU to train on Llama 3 (8B), which delivers reasonable speeds.

We are also training one on FLAN-T5-Base by Google, which is a text-to-text transformer that is pretrained on certain things. We will be fine tuning this. It has 250M parameters, which is significantly smaller than Llama 3's 8B. We are looking towards this as a more lightweight solution that still delivers some accuracy.

## Validation methods
We will look at the score the tuned model is able to achieve on We will evaluate model performance using:
- **Quantitative metrics:** Standardized test scores on benchmark datasets.


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

Originally, the FLAN-T5 was able to correctly classify and output the outside nested statement after 4 iterations through the 30k, when focusing on just the outer one. We did this to check if it was learning and it is. In 96% of test case samples it got the first operator right when predicting for just one operator, which means it is learning somewhat. However, by expanding the labels to their full equations we noticed it is not learning as well for unknown reason.

Currently, FLAN-T5 is outputting an average of 78% similarity to the correct formulas. About 20% of examples are entirely correct, and another 20% are just missing some formatting that can be fixed easily (mostly missing ending parenthesis). We estimate that with FLAN-T5-Base, 40% of these problems should output correct answers. We are also looking to try FLAN-T5-Large, which is almost 800M parameters.


## Basic performance metrics
The A100 is faster than the T4 which we used previously, which has helped us massively. An epoch of training takes about 15 minutes on FLAN-T5-Base.

## Test case results
We observed that the rewards started increasing very slowly at first, but eventually picked up more steam. Average-of-5 rewards for steps seem to be going up steadily as the base model trains.

## Current limitations
We think that FLAN-T5-Base may have hit its limit on our data, as training additional epochs leads to little improvement. It often predicts the <eos> token early, causing early truncation. For most examples (up to 70-80%), the outputs are entirely correct up to the point they suddenly cut off. We are looking into what causes this behavior, as it is not something obvious like a generation limit.

## Resource usage measurements
Training eats up a significant chunk of the A100's VRAM, but it is not maxed out meaning there is overhead somewhere else. Training will not go faster than this.

## Unexpected challenges
As explained in current limitations, we are dealing with an early truncation issue in many of our test examples.

# Next Steps

## Immediate improvements needed
1. Improve training to get the *entire* formula
2. Expand Literature Review


## Technical challenges to address
We need to nail down a balance between a numerical metric and a human-based one and weigh those properly to judge reasoning performance. Since this model will be used to directly guide humans, it needs to output satisfactory if not great results that humans can understand.

We also need to attempt at balancing training time with reasoning quality, to find the right trade-off between training speed and performance.


## Questions you need help with
Is there any ways to force a transformer to output more, even if the loss gets worse?

## Alternative approaches to try
There are many base models that can be experimented with, with differing logics and parameter sizes. We could look into some of those if we find that training is too slow or the bigger model is overkill. Additionally, we have considered looking into the actual training process and optimizing that instead of results.

As of now, we have pivoted to experimenting with transformer-based approaches.

## What you've learned so far
We have learned a lot about GRPO and reasoning models as this is our groups' first time looking into them in depth. We are also learning a lot about transformers and their training methods.

# Literature Review

Our literature review examined several sources that contributed to our understanding of GRPO models and reinforcement learning techniques applied to reasoning tasks.

- **DeepSeek-R1 (DeepSeek-AI, 2025)** demonstrated how reinforcement learning with GRPO can significantly improve the reasoning capabilities of LLMs. The study found that reward-based fine-tuning improved the model’s ability to solve math, coding, and standardized test questions without requiring extensive human annotation. This validates our choice of GRPO as a reinforcement learning framework for our project.

- **DeepSeekMath (Shao et al., 2024)** introduced GRPO as a variation of PPO, demonstrating how it can be used to train math reasoning models efficiently. The key takeaway from this paper was that GRPO avoids the need for a critic network by using group-based baseline rewards, which could significantly reduce memory and computational costs in our project.

- **Math-Shepherd (Wang et al., 2024)** proposed a reward model that improved step-by-step verification of solutions in math problem-solving tasks. This highlighted the importance of reward shaping in RL-based reasoning models, which we plan to incorporate into our own fine-tuning process.

- **Setlur et al. (2024)** explored reinforcement learning on incorrect synthetic data, showing that fine-tuning on negative samples dramatically increased training efficiency. This suggests that our training data can be augmented with incorrect answers to reinforce model learning, making training more sample-efficient.
