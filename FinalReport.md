# Problem Statement

## Our Optimization

Our group is looking to optimize various models (including Unsloth-based GRPO and T5 Transformer) to be able to output the answer when given American high school level, SAT-style mathematics word problems. Specifically we want to optimize for the highest performance on test banks and as an intermediate step, closeness formulas to the actual formula that is used to solve the question.


## Problem Motivation

The ultimate goal of attacking this problem is to be able to have a model that can be used as a study tool, to help students prepare for the SAT/ACT, other standardized exams, or just mathematics exams in general. Many students struggle with preparation; tutors are expensive and books are inflexible. A specially tuned AI chat-bot would be able to overcome both of these constraints and help students achieve higher scores with less struggle. Getting the model to be able to get to the right answer is the first step in this process.



## Measuring Success

We have two metrics for measuring success. The first is the Normalized-Levenshtein Score (NLS), which is a function of edit distance compared to string length. The second is accuracy on a test set. We use both because one is the end goal and the other is a good comparison metric. Completely accurate answers are a lot more difficult to come by than a similar formula, as even 1 wrong number won’t affect the formula too much but will cause the answer to be inaccurate. Therefore, the similarity score gives us more granularity on what the model is really doing; two models could have the same accuracy but if one is getting questions wrong by missing a number and the other is getting questions wrong by using the wrong formula entirely, the NLS will differentiate the two.



## Constraints

Our main constraint is the data. While it has a fairly large amount of examples for smaller models, it is not a lot in the grand scheme of data. The quality of it is also questionable, as it contains many grammar mistakes, incorrect answers/formulas pairings, and it’s formatted inconsistently. We did our best to remedy these issues, but it’s likely that some of these issues persist. Nonetheless, the dataset is the only one we found that has formulas to go with the questions, which was a big help during training.



## Data

Our process requires us essentially three elements: the word problem, the formula, and the answer. In our particular dataset, the answers came as a lettered list of options and a correct answer. We got the numerical answer by extracting the correct answer from the list and removing any irrelevant formatting. We basically need large numbers of these question-formula-answer triplets. The data we used had other fields as well, but we did not use them for this project.



## Points of Contention

This was a concern from the start, but the issue of exactness popped up again and again. There are often more than one ways to solve a mathematics problem, but once you choose a certain way the formula must be exact. Therefore, models that “kind of” understand what is going on are a problem because they can identify basic functions (like needing to use divide), but they’re not exact enough to get all the numbers right or have them in the right order. This causes a much higher NLS than accuracy, with the magnitude of the difference depending on how the model is trained.



# Technical Approach

## Mathematical Formulation

Our objective functions are twofold. Therefore first one is the percentage of answers it gets completely correct, which is the ultimate goal. The second is how close the model outputs are to the ground truth, even if they are not exactly correct (NLS)

Specifically, we are looking at:

$$
\M^* = \arg\max_{\M} M(E) = E^*
$$

where M(E) is the output of the model on problem E and E* is the ground truth as well as

$$
\M^* = \arg\max_{\M} NLS(M(E), E^*)
$$

where M(E) and E* are the same as above, and the NLS() function measures the similarity ratio between the outputs.


## Algorithm/Approach Choice and Justification

The initial approach we took was Gradient-based Policy Optimization (GRPO), a policy gradient method that improves the model’s reasoning ability via reinforcement learning (RL). GRPO can optimize multi-step reasoning paths, meaning it can capture long-range dependencies. It’s a reinforcement-learning based model that uses rewards to tune weights over time, following methods explored in DeepSeekMath (Shao et al., 2024). GRPO gradually improves response distribution through sampled trials, which means it’s less likely to be overconfident. The GRPO method we used is based on a relatively new framework called Unsloth, which essentially significantly reduces the VRAM needed to train these huge LLM models like Llama, making them possible to train on single-GPU systems or even Google Colab.

We also used a transformer-based architecture called T5. Transformers were used because, like GRPO, they can also capture long-range dependencies. This ability is valuable because math problems often require the entire context, and if a model just looks at a piece it will miss an important detail. However, transformers stood out because a lot of them can be made quite small, unlike the LLM-based models we were using. For example, the model we ended up going with called FLAN-T5 has just 250M parameters compared to the 8B with the Llama model. This results in it being significantly lighter, quicker to train, and more suited to our relatively small dataset. We chose FLAN-T5 because its instruction-tuned and thus will have a bit of a strong base to start on.



## Implementation Strategy

At first, we just used the data to try to train for the answer using the Unsloth framework with GRPO. We used a couple different reward functions including closeness to the answer (numerically). However, this doesn’t get down to the crux of the issue which is understanding how the word problem relates to the process, and how the process relates to the answer. Thus, the model wasn’t learning anything. 

Reviewing the literature, we decided to switch over to a symbol-based approach, trying to predict the formula instead of the answer. There were a number of preparations we needed to make to do this. First, we cleaned the data by extracting the numerical answer from each row. Then, we tried to process the data by fixing the formatting to be consistent as well as adding some of our own formatting (such as spacing). Finally, we used the regex/Sympy parser to filter out questions in which the listed answer does not match the formula. We did this to help limit the errors the model could make, like concatenating ‘(‘ to the end of each operator since an operator is always followed by a ‘(‘. We then used this new data with the NLS to help train the GRPO Reinforcement Learning model. After realizing that the Llama model might be a little large for our dataset, we swapped over to the transformer model with cross-entropy loss on each token - essentially, how confident was the model on each particular token and was it correct or not. We also trained on smaller (not necessarily simpler) questions, to reduce the size of the transformer’s matrix. The quicker training speed of the smaller transformer models allowed us to play with the hyperparameters more.

Once the output formula is obtained, we use regex to put it in symbolic form and then call Sympy to turn it into a final answer.


## Validation Methods

We check the accuracy of the models on a sample of a test set, essentially creating an SAT-type exam out of our test set. We also calculate the average NLS across the sample to compare models with each other.

## Resource Requirements and Constraints

The GRPO model had some pretty stiff resource requirements. Unsloth’s framework allows models to be trained on a single GPU, but it still takes up a vast majority of that GPU. Even on Colab’s A100 GPU with 40 GB of available VRAM, it takes up most of the GPU and is still quite slow. 

The transformer model, due to its small size, was much more flexible. Batch size could be adjusted to fit the GPU, with the A100 being able to accept 32 with 2 accumulation steps without running out of memory.

Overall, the speed was the constraint, not the memory.


# Results

## Evidence Implementation Works

The initial results on the GRPO model directly predicting the answer were not good. Even after multiple hours of training, it didn’t learn anything and it felt like it was just randomly guessing.

The GRPO model predicting the formula faired a little better. It was able to start learning the context behind math problems and even started to get select few questions correct. 

We started our FLAN-T5 exploration and experimentation by doing a proof of concept model: just predicting the outermost operator based on the question. We did this by truncating answers down to a single token (since they always open with an operator). This is a relatively a simple 4-class classification task, and the initial results from the model were extremely good.

Finally, our FLAN-T5 model gave us the best results. It was able to pretty consistently get a large majority of the formula, and its accuracy and NLS were far superior to earlier models (will be discussed under “Basic Performance Metrics”).


## Basic Performance Metrics

The number-predicting GRPO model essentially had an accuracy of 0. As a side note, we mistakenly displayed incorrect accuracies in the final presentation for GRPO as there were a couple of crucial mistakes made in the calculations. The NLS scores, however, remain the same. The formula-predicting GRPO model did better, as it actually seemed to learn the context behind the questions. It’s NLS averaged out to about 0.50 and the accuracy was significantly lower at around 0.1 (both have baseline ~0). This goes back to the issue of exactness - with many tokens in the answer, and an average NLS of 0.50, the chance that all the tokens are completely correct is very slim. Our final FLAN-T5 model performed better. Its average NLS was >0.80, and the accuracy was 0.72 (baselines of ~0.2 and ~0). The results show that the FLAN-T5 model was making the right connections between word problem and formula.


## Test Case Results

Our results fell into 4 main categories. The first screenshot shows examples where the model outputs the correct answer, and the second screenshot shows examples of when it doesn't.

<img width="754" alt="Screenshot 2025-05-12 at 6 03 17 PM" src="https://github.com/user-attachments/assets/fca31c07-4e86-4f9e-87a8-a704f5c43e6c" />

<img width="754" alt="Screenshot 2025-05-12 at 6 03 21 PM" src="https://github.com/user-attachments/assets/32bc940b-a776-401c-bc4f-c47ced82f774" />

## Current Limitations

The GRPO model’s limitations mostly have to do with training speed. Even using the A100, a single step takes almost a minute and you need multiple hundreds of steps to see anything at all. This is an issue because each hour on an A100 is roughly 7.5 compute units, so each training cycle is about $4. There is a lot of experimentation to be done with parameters and it wasn’t feasible with the slow speed of training the GRPO model. We also tried the T4 which has negligible cost, but the training times more than doubled compared to A100 which is even more infeasible.

FLAN-T5 alleviates this a bit. It’s trainable on T4, although the decreased VRAM means the batch size must be reduced when compared to A100. On T4, a typical training cycle takes roughly an hour and change, meaning that trying different hyperparameters is a viable strategy. With A100, the training cycle is not only reasonable but quite quick, with a full cycle being able to be completed in roughly half an hour. The limitations with FLAN-T5 seem to be more with the power of the model and the training data, and less with the ability of the model to train. 


## Resource Usage Measurements

For the GRPO model, VRAM is high but not maxed out on the default settings (between 20 and 35 GB).

By adjusting the batch size for the transformer model, the VRAM can be brought fairly high. On the A100, VRAM usage generally sits between 25 and 30 GB on batches of size 32 with 2 gradient accumulation steps. This can be toned down to fit on a T4 without issue, albeit at a slower training speed.


## Unexpected challenges

The first unexpected challenge we had was trying to get a reward function for the GRPO model that was trying to just predict the right answer. Also, the training time was an unexpected challenge. We knew it would take a while, but we figured that with relatively short inputs and outputs it wouldn’t be too excessive. 

For the GRPO model predicting formulas, the main unexpected challenge we faced was the issue of repeated tokens and no end of sequence (EOS) token. When this happened the model would generate part of a formula, and then it would just repeat a token until the artificial limit (most commonly ‘multiply’). 

The T5 model was a lot smoother, but there were still some unexpected challenges. One was getting the training loop set up, as it comes with a lot of different parts. Its NLS score average was quite high, but the accuracy was a little lower. This reflects a model which is getting the general idea, but lacks the weighting to be exact. 

# Next Steps

## Immediate Improvements

At the current stage of the project, there are not really immediate improvements that need to be made. Improvements at this point are likely long-term problems that have many different steps.

## Current Technical Challenges, Questions, and Alternative Approaches

For future work, going back and taking a closer look at the data could be a good idea. With the size of the dataset, it will take an unreasonable amount of time to clean by hand. We did some basic cleaning, but perhaps with more time we could create automated systems that can quickly fix a lot of the issues with the data. At this point, we think that the data quality is hampering the generalizability of the model a little bit, especially as far as grammar is concerned. 

Another technical challenge to be addressed has to do with exactness discussed earlier. We are curious as to whether the last 20% missing in the average NLS score is due to model complexity or training parameters. There are two other versions of reasonably sized FLAN-T5 models, those being small (80M) and large (780M). It would be interesting to compare the overall performance of those models with the base (250M).

## What We Learned

This project was a valuable experience for all of our group. As a collective team, we had not really worked with this type of Machine Learning before, and especially not with optimizing text-to-text transformers and LLMs. Specifically, we learned about GRPO as a concept as well as how tuning different parameters and loss functions affects model training. The final NLS did get quite high and the accuracy was decent, showing that our work worked. Overall, this project was a very rewarding experience for all of us.




# Literature Review
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

# Bridging Literature and Our Work

## Introduction

Our approaches, while simpler than some of the more well-resourced approaches found in the academic literature, can be more comprehensively understood in the context of this literature. While the following section is by no means extensive, it represents the literature we have identified most related to our approaches in terms of initial approach, results, decisions for further approaches, etc.


## Approaches

Broadly, we took three approaches to this project. The first was a direct-solve approach through reasoning and GRPO. The second was similar, but instead of focusing on solving the problems directly, we optimized for a semantic parsing of the MWP, turning it into a formula. The final approach was our FLAN-T5, which was the best-performing method of the three.


## Symbolic Solvers

The first major change in approach was from a direct solve approach to a semantic parsing approach. One paper that gave us insight, in hindsight, into the utility of this decision was Hosseini et al. (2014). As discussed above, this 2014 paper's approach called ARIS was able to achieve 77.7% accuracy on primary school word problems, similar in scale and difficulty to the ones in our data set.

This approach was, at least conceptually, closest to our final approach with FLAN-T5. This is since our first approach was direct, so did not focus on categorization and parsing, and subsequent approaches used parsing and then a simpler model. ARIS, a pre-transformer method, was surprisingly accurate while taking a purer NLP approach. As later discussed, much of the error in the Llama approaches could be attributed to hallucinations in intermediate steps.

Another paper that gave insight into the strength of switching to a symbolic solver approach was He-Yueya et al. (2023). That paper showed a 20% gain in performance on the ALGEBRA benchmark using declarative and incremental representations interfacing with external tools rather than simple Program-Aided Language model approaches. 

The final insight into this approach we identified in the literature was in Sprague et al. (2025)
![Screenshot 2025-05-13 at 5 05 28 PM](https://github.com/user-attachments/assets/198edc0c-7635-400f-8c34-0ff5f960f2f1)
This chart shows performance gains across the board using tool solvers on the Math datasets. Additionally, it shows the relative weakness of purely a direct approach, results mirrored in the change in our outcomes throughout our approaches.

In our case, however, the switch from the first to the second approach did not see a large gain in performance. He-Yueya et al. (2023) notes that prior work that has experimented on using LLMs to generate equations and solve them with external solvers "generally improves final performance by less than 5% on GSM8K" (p2). So, this paper shows that our gain in performance was in line with literature, and that its overall small effect also was in line with previous research.

## Chain of Thought
The second major area of focus in analyzing connections between the results we found and the existing literature on our topic concerns CoT approaches. Sprague et al. (2025), notably, finds the places where CoT helps more and less.

Sprague et al. (2025) outlines the following major findings: "Finding 1: CoT only helps substantially on problems requiring mathematical, logical, or algorithmic reasoning," and "Finding 2: CoT primarily helps with the execution step that performs computation and symbolic manipulation, but falls short of what LLMs with tool augmentation can do" (p. 2).
Additionally, they note that the results "paint a picture that CoT's utility is often circumscribed by tool augmentation: on problems where CoT helps, we already have more powerful tools than CoT that we can employ" (p. 2).

These findings paint a picture that can help explain why CoT was not very successful in our case. Our problems were relatively simple and required little reasoning, so we hypothesize that any reasoning gains using CoT may have been counteracted by hallucinations in intermediate steps in the reasoning process, along with a myriad of other possible issues that a CoT approach brings. 


## Model Size

One finding that could help explain the results for CoT is found in Kojima et al. (2023).
![Screenshot 2025-05-13 at 5 05 38 PM](https://github.com/user-attachments/assets/ccdf2ef0-6bb1-4e84-8a71-51e4ce039e2b)

The chart above shows model performance in terms of model size and CoT. It finds that CoT benefits kick in above the 8B parameter range. Notably, our Llama model we used was 8B parameters. While the scope and focus of our project and the above research is not identical, it is an interesting finding nonetheless. Perhaps future research will find optimal use-cases for CoT in terms of model size, application, and other parameters.

## Problem Difficulty

Another consideration in identifying how our process compared to the existing body of literature pertains to the types of problems. The graphic below, adapted from Ma et al. (2025), shows model performance with token limits on three Math problem data sets, easier on the left and harder on the right.
![Screenshot 2025-05-13 at 5 05 45 PM](https://github.com/user-attachments/assets/6063a48c-4013-42aa-a04f-2e307204dbc1)

The problems contained in our dataset were easier than the above problems, but their findings can highlight some elements of our findings. The chart illustrates performance differences between their "NoThinking" and "Thinking" approaches. NoThinking bypasses the CoT reasoning in reasoning models and has it directly output the answer. This prompt-engineering technique shows considerable performance advantages in a token-limited setting. Given that our limit was just ~2k tokens, CoT could have hurt our results more than it helped. 
Another interesting finding is that of problem difficulty. While there are no major consistent patterns in this domain, looking at our results in the context of being closer to the AMC than the other two benchmarks gives us context to compare our findings to those of other approaches. 

## Conclusion
Overall, many facets of our approaches have connections with findings in the literature we explored. To connect some of the above ideas, it is important to note that our final approach, FLAN-T5, has 250M parameters, compared to Llama's 8B. Using a model with 3% of the parameters yielded better results. Through this analysis, we hypothesize that this is mainly due to CoT not producing measurable benefits in smaller models, the token limits making a more direct parsing method superior, and the problem difficulty. Given that, in 2014 Hosseini et al. (2014) was able to implement a very successful but much simpler approach to achieve a similar goal as us, it is reasonable to conclude that a CoT approach would have made more sense for a harder problem set, not the one we used.


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

