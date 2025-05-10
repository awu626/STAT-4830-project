


# Introduction
The use of Machine Learning (ML) techniques to solve math problems presents several avenues for significant improvement in various domains. On the one hand, evaluating model performance on datasets with verified answers presents a straightforward, labeled avenue for improving model performance on problem-solving and reasoning tasks. On the other hand, or more accurately on the other extreme, large language model (LLM) usage in the field of automated proof verification (and possibly generation) has the potential to revolutionize Mathematical research. 

More broadly, as LLMs become more widely used throughout both academic circles and the general public, the classification and concept of 'intelligence' will likely change. In the same way that standardized testing has historically been used as a sort of human 'benchmark,' it is being used to train and improve the problem-solving intelligence of LLMs. For this reason, we chose to focus on optimization in the field of using models to solve math word problems.

Given the immense variety in possible types of math problems, the approaches in optimizing solving those problems differ drastically. The scope of this review pertains to datasets that contain math word problems (MWPs) with at least a question and an answer. (Ahn et al. 2024) Presents a useful framework for categorizing the types of problems, and later extending to the types of approaches. 

## Question Types and Datasets
**Question-Answer (QA)** MWPs contain a question in word form and an answer. This answer is often solely numeric, but can be a short answer of another type. Additionally, QA answers are either just the answer itself, or, the possible answer choices, along with the correct answer. This multiple-choice data is often from multiple-choice standardized tests.
- $\mathcal{Q}$: Beth bakes 4, or 2 dozen batches of cookies in a week. If these cookies are shared amongst 16 people equally, how many cookies does each person consume? 
- $\mathcal{A}$: 6

**Question-Equation-Answer (QEA)** MWPs include a word-problem question and the mathematical equation that, when evaluated, deterministically solves for the correct answer. Some of these datasets could more accurately be described as QE, as they do not contain the answer explicitly. 
- **Note: The fact that they don't contain the answer doesn't really matter. Several studies and approahces have concluded that, rather than solving directly for an answer, LLM's strengths in the realm of simpler MPW solving are in parsing out the formula out of the word problem, rather than solving for the mathematical answer directly.**
- $\mathcal{Q}$: Beth bakes 4, or 2 dozen batches of cookies in a week. If these cookies are shared amongst 16 people equally, how many cookies does each person consume? 
- $\mathcal{E}$: ( ( 4 * 2) * 12 ) / 16
- $\mathcal{A}$: 6

**Question-Rationale-Answer (QRA)** MWP datasets are the final type within the scope of this review. These, like the name implies, give the question, the rationale behind the answer, and the answer itself. 
- $\mathcal{Q}$: Beth bakes 4, or 2 dozen batches of cookies in a week. If these cookies are shared amongst 16 people equally, how many cookies does each person consume? 
- $\mathcal{R}$: Beth bakes 4 2 dozen batches of cookies for a total of 4 ∗ 2 =<< 4 ∗ 2 = 8 >> 8 dozen cookies. There are 12 cookies in a dozen and she makes 8 dozen cookies for a total of 12∗8 =<< 12∗8 = 96 >> 96 cookies. She splits the 96 cookies equally amongst 16 people so they each eat 96/16 =<< 96/16 = 6 >> 6 cookies. 
- $\mathcal{A}$: 6

## Pre-Transformer Approaches
### Overview
While all of the details and intricacies pertaining to problem-solving strategies and tools do not solely come from the three categories outlined above, the vast majority can be applied to these sorts of problems, or offer insight into technique successes on certain types of problems and failure in others. Most approaches to using LLMs for mathematical problem solving use a wide variety of techniques, so this overview of approaches aims to articulate the differentiating features of approaches. 
Additionally, the approaches we outline are not close to a comprehensive overview of all that exists. Rather, we aim to provide a context and foundation in optimization literature through which to analyze our project, choices, outcomes, and thought process.

### Approaches (Pre-Transformer)
Our project, and much of the literature review, focuses on transformer enabled MWP solving strategies. However, an overview of the pre-transformer approaches gives insight into the more novel techniques. 
Zaporojets et al. (2021) conveniently categorizes approaches in solving arithmetic word problems into the following main categories: (i) Rule-based systems, (ii) Statistical systems, and (iii) Neural network systems.

I. Rule-based systems
The first important rule-based approach is found in (Bobrow 1964). Specifically, STUDENT is a rule-based parsing program that turns word problems into number values and operations. Notably, this lays the groundwork for graphical approaches in the field, as the values outlined in the problem are nodes and the relationships between those values, the calculations, are edges.
Later extensions of this work implemented more rules and more specific relationships, like those of speed rates (like km/h) or groups of items (like a dozen meaning twelve).

II. Statistical systems
Statistical systems (often) similarly parse out the problem into a tree representation, but in a different way. Statistical systems capture patterns in word-problem data and attempts to predict nodes and edges through that.
One paper that inspired our thinking was Hosseini et al. (2014). This paper cleverly uses verb categorization to solve simple arithmetic WPs. This method, called ARIS, is able to categorize verbs with 81.2% accuracy and solves 77.7% of elementary school math problems.

III. Neural network systems
Neural network systems, naturally, extend form the previous two approaches in the realm of neural networks. These are much closer in approach to transformer-enabled LLM solving.


## LLM/Transformer-Based Approaches
### Overview

**DO THIS**


### Chain of Thought
Among the most clear categorizing factors for MWP solving approaches is whether or not the strategy implements Chain of Thought (CoT). Using this approach, LLMs 'reason' by producing intermediate reasoning steps instead of outputting a final answer. 

Wei et al. (2023) demonstrated the ability for LLMs to reason through chain-of-thought prompting. This unlocked significant development in implementing LLMs for problem solving, along with other advances in the field. Notably, Wei et al. (2022) implemented chain-of-thought for reasoning on the GSM8K (Cobbe et al. 2021) dataset, one of the most widely used data sets for elementary school level MWPs. 

### Process/Outcome-Based Supervision
Uesato et al. (2022) introduces the notion of separating process and outcome-based supervision in the context of chain-of-thought. 
Outcome-based supervision refers to training methods that evaluate only model correctness. Process-based supervision, however, evaluates intermediate steps in the chain-of-thought reasoning process for correctness and includes errors pertaining to these, in addition to final answer correctness.
This distinction can be in the context of a policy model or the more general reasoning model. In both cases, the distinction generally refers to grading intermediate reasoning steps for correctness versus outcome-only. Process-based supervision has a number of possibilities in terms of implementation strategy. Most often, a QRA dataset is used to train the model to reason intermediate steps between the question and answer. However, especially in applications outside the simpler MWP domain, other strategies are used for this purpose.
Uesato et al. (2022) finds performance gains with process-based supervision, improving final answer error rate from 16.8% to 12.7% by using this technique. Lightman et al. (2023) focuses instead on training only a policy model, without tuning the generative CoT model itself, and obtains similar results.


### Multi-Step Planning

**DO THIS**

### External Solvers

**DO THIS**

# Bridging Literature and Our Work
## Introduction
Our approaches, while simpler than some of the more well-resourced approaches found in the academic literature, can be more comprehensively understood in the context of this literature. While the following section is by no means extensive, it represents the literature we have identified most related to our approaches in terms of initial approach, results, decisions for further approaches, etc. 


Notes
- CoT or Not
- Zero or Multi-Shot CoT
- Tool-Augmented
- Thinking or NoThinking
	- Ma et al. 2025

- Symbolic Solvers
	- He-Yueya et al. 2023
	- Sprague et al. 2024
- Model Size and CoT
	- Kojima and Gu, 2022

- Token Limits and Question Difficulty


# References - Not Formatted
[TO COT OR NOT TO COT? CHAIN-OF-THOUGHT HELPS MAINLY ON MATH AND SYMBOLIC REASONING](https://arxiv.org/abs/2409.12183v1)
- T2: Why CoT may not have been the best for this
- CoT benefits in different kinds of tasks
- p5 Top

[Solving Math Word Problems by Combining Language Models With Symbolic Solvers](https://arxiv.org/abs/2304.09102)
- He-Yueya et al. 2023
- T1
- Performance gains in combining LLM and Symbolic Solver

[Insights into Pre-training via Simpler Synthetic Tasks](https://arxiv.org/abs/2206.10139)
- Pre-training frameworks like LIME

[A Survey of Deep Learning for Mathematical Reasoning](https://arxiv.org/abs/2212.10535)
- Great for overarching Lit Review
[Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)

[Learning to Solve Arithmetic Word Problems with Verb Categorization](https://aclanthology.org/D14-1058/)
- Hosseini et al. 2014
- T1&2
- Semantic parsing 
- 2014 paper

[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://www.alphaxiv.org/abs/2504.11536)

[SymbolicAI: A framework for logic-based approaches combining generative models and solvers](https://www.alphaxiv.org/html/2402.00854v1)

[LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning](https://arxiv.org/pdf/2101.06223)
- Pretraining for math reasoning benchmarks


[Large Language Models for Mathematical Reasoning: Progresses and Challenges](https://www.alphaxiv.org/html/2402.00157v1)
- (Ahn et al 2024)
- Great Source
- Types of problems

[Knowledge Augmented Complex Problem Solving with Large Language Models: A Survey](https://www.alphaxiv.org/html/2505.03418v1)

[Large Language Models and Mathematical Reasoning Failures](https://www.alphaxiv.org/html/2502.11574v1)

[A Survey on Mathematical Reasoning and Optimization with Large Language Models](https://www.alphaxiv.org/pdf/2503.17726)

[Mathify: Evaluating Large Language Models on Mathematical Problem Solving Tasks](https://www.alphaxiv.org/html/2404.13099v1)

[Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858)
- NoThinking vs Thinking
- Ma et al. 2025




[Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Invented transformer
- (Vaswani et al. 2017)

History of Solving Methods
- [Solving Arithmetic Word Problems by Scoring Equations with Recursive Neural Networks](https://arxiv.org/abs/2009.05639)
	- Zaporojets et al. 2021
- Bobrow, D. G. (1964). Natural language input for a computer problem solving system.


[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- Wei et al. 2022

[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- Cobbe et al. 2021
- GSM8K

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- Lightman et al. 2023
- OpenAI

[Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275)
- Uesato et al. 2022
- Process and outcome-based feedback

