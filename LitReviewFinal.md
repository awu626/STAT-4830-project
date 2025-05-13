


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

Transformer based large language models have become the central template to solving math word problems, due to their strong performance of a range of natural language processing tasks. Unlike earlier neural architectures, transformers are capable of more effective long-range reasoning, generalisation over symbolic inputs, and in-context learning capacities that allow them to quickly fit to new problem structures. As outlined in Lu et al. (2022), this shift has expanded the scope of mathematical reasoning tasks models can compute, from basic arithmetic to theorem proving and symbolic manipulation. LLM-based approaches often vary in their model scale and in their strategies, for example, whether reasoning is implicit or step-by-step, if external tools are used, and the kind of supervision is used for training. 


### Chain of Thought
Among the most clear categorizing factors for MWP solving approaches is whether or not the strategy implements Chain of Thought (CoT). Using this approach, LLMs 'reason' by producing intermediate reasoning steps instead of outputting a final answer. 

Wei et al. (2023) demonstrated the ability for LLMs to reason through chain-of-thought prompting. This unlocked significant development in implementing LLMs for problem solving, along with other advances in the field. Notably, Wei et al. (2022) implemented chain-of-thought for reasoning on the GSM8K (Cobbe et al. 2021) dataset, one of the most widely used data sets for elementary school level MWPs. 

### Process/Outcome-Based Supervision
Uesato et al. (2022) introduces the notion of separating process and outcome-based supervision in the context of chain-of-thought. 
Outcome-based supervision refers to training methods that evaluate only model correctness. Process-based supervision, however, evaluates intermediate steps in the chain-of-thought reasoning process for correctness and includes errors pertaining to these, in addition to final answer correctness.
This distinction can be in the context of a policy model or the more general reasoning model. In both cases, the distinction generally refers to grading intermediate reasoning steps for correctness versus outcome-only. Process-based supervision has a number of possibilities in terms of implementation strategy. Most often, a QRA dataset is used to train the model to reason intermediate steps between the question and answer. However, especially in applications outside the simpler MWP domain, other strategies are used for this purpose.
Uesato et al. (2022) finds performance gains with process-based supervision, improving final answer error rate from 16.8% to 12.7% by using this technique. Lightman et al. (2023) focuses instead on training only a policy model, without tuning the generative CoT model itself, and obtains similar results.


### Multi-Step Planning

Solving complex math problems often requires decomposing them into manageable sub-tasks, something that humans use naturally, but models must learn explicitly. In LLMs, multi-step planning is normally implemented through structured prompting strategies or architectural scaffolding that encourages decomposition of the problem. One representative method is Least-to-Most Prompting (Zhou et al., 2023), where models are prompted to reduce the problem into smaller sub-problems and then solve them sequentially. This will often improve correctness by making these intermediate reasoning steps more explicit and less susceptible to error. 

Other techniques include program-of-thoughts prompting (Chen et al., 2022), where reasoning is expressed as pseudocode or executable steps, and then performed outside the model. These planning techniques help reduce common pitfalls like shortcut reasoning or incorrect assumptions that occur when the model tries to complete the problem in a single step.  

This is especially useful in domains where structured reasoning and accuracy matter, like geometry and symbolic logic. Multi-step approaches also often integrate well with process-based supervision techniques, as each step in the plan can be independently assessed and optimised. 


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


## References

1. Z. Sprague, F. Yin, et al., [To CoT or Not To CoT? Chain-of-Thought Helps Mainly on Math and Symbolic Reasoning](https://arxiv.org/abs/2409.12183v1), arXiv preprint arXiv:2409.12183v1, 2024.

2. J. He-Yueya, G. Poesia, et al., [Solving Math Word Problems by Combining Language Models With Symbolic Solvers](https://arxiv.org/abs/2304.09102), arXiv preprint arXiv:2304.09102, 2023.

3. Y. Wu, F. Li, et al., [Insights into Pre-training via Simpler Synthetic Tasks](https://arxiv.org/abs/2206.10139), arXiv preprint arXiv:2206.10139, 2022.

4. P. Liu, L. Qiu, et al., [A Survey of Deep Learning for Mathematical Reasoning](https://arxiv.org/abs/2212.10535), arXiv preprint arXiv:2212.10535, 2023.

5. T. Kojima, S. Gu, et al., [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916), arXiv preprint arXiv:2205.11916, 2023.

6. M. J. Hosseini, H. Hajishirzi, O. Etzioni, and N. Kushman, [Learning to Solve Arithmetic Word Problems with Verb Categorization](https://aclanthology.org/D14-1058/), in *Proc. EMNLP*, 2014.

7. J. Feng, S. Huang, et al., [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://www.alphaxiv.org/abs/2504.11536), AlphaXiv preprint, 2025.

8. M. Dinu, C. Leoveanu-Condrei, et al., [SymbolicAI: A Framework for Logic-Based Approaches Combining Generative Models and Solvers](https://www.alphaxiv.org/html/2402.00854v1), AlphaXiv preprint, 2024.

9. Y. Wu, M. Rabe, et al., *LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning*, 2022.

10. J. Ahn, R. Verma, et al., [Large Language Models for Mathematical Reasoning: Progresses and Challenges](https://www.alphaxiv.org/html/2402.00157v1), AlphaXiv preprint, 2024.

11. D. Zheng, L. Du, et al., [Knowledge Augmented Complex Problem Solving with Large Language Models: A Survey](https://www.alphaxiv.org/html/2505.03418v1), AlphaXiv preprint, 2025.

12. J. Boye, B. Moelle, [Large Language Models and Mathematical Reasoning Failures](https://www.alphaxiv.org/html/2502.11574v1), AlphaXiv preprint, 2025.

13. B. Forootani, [A Survey on Mathematical Reasoning and Optimization with Large Language Models](https://www.alphaxiv.org/pdf/2503.17726), AlphaXiv preprint, 2025.

14. A. Anand, et al., [Mathify: Evaluating Large Language Models on Mathematical Problem Solving Tasks](https://www.alphaxiv.org/html/2404.13099v1), AlphaXiv preprint, 2024.

15. Q. Ma, et al., [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858), arXiv preprint arXiv:2504.09858, 2025.

16. A. Vaswani, et al., [Attention is All You Need](https://arxiv.org/abs/1706.03762), arXiv preprint arXiv:1706.03762, 2017.

17. K. Zaporojets, et al., [Solving Arithmetic Word Problems by Scoring Equations with Recursive Neural Networks](https://arxiv.org/abs/2009.05639), arXiv preprint arXiv:2009.05639, 2021.

18. D. G. Bobrow, *Natural Language Input for a Computer Problem Solving System*, RAND Corporation, 1964.

19. J. Wei, et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), arXiv preprint arXiv:2201.11903, 2022.

20. K. Cobbe, et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168), arXiv preprint arXiv:2110.14168, 2021.

21. E. Lightman, et al., [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050), arXiv preprint arXiv:2305.20050, 2023.

22. OpenAI, *GSM8K*, 2021.

23. J. Uesato, et al., [Solving Math Word Problems with Process- and Outcome-Based Feedback](https://arxiv.org/abs/2211.14275), arXiv preprint arXiv:2211.14275, 2022.
