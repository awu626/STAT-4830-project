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



## Specifically, we are looking at:

FORMULA

where M(E) is the output of the model on problem E and E* is the ground truth as well as

FORMULA

where M(E) and E* are the same as above, and the SIM() function measures the similarity ratio between the outputs.


## Algorithm/Approach Choice and Justification

The initial approach we took was Gradient-based Policy Optimization (GRPO), a policy gradient method that improves the model’s reasoning ability via reinforcement learning (RL). GRPO can optimize multi-step reasoning paths, meaning it can capture long-range dependencies. It’s a reinforcement-learning based model that uses rewards to tune weights over time, following methods explored in DeepSeekMath (Shao et al., 2024). GRPO gradually improves response distribution through sampled trials, which means it’s less likely to be overconfident. The GRPO method we used is based on a relatively new framework called Unsloth, which essentially significantly reduces the VRAM needed to train these huge LLM models like Llama, making them possible to train on single-GPU systems or even Google Colab.

We also used a transformer-based architecture called T5. Transformers were used because, like GRPO, they can also capture long-range dependencies. This ability is valuable because math problems often require the entire context, and if a model just looks at a piece it will miss an important detail. However, transformers stood out because a lot of them can be made quite small, unlike the LLM-based models we were using. For example, the model we ended up going with called FLAN-T5 has just 250M parameters compared to the 8B with the Llama model. This results in it being significantly lighter, quicker to train, and more suited to our relatively small dataset. We chose FLAN-T5 because its instruction-tuned and thus will have a bit of a strong base to start on.



## Implementation Strategy

At first, we just used the data to try to train for the answer using the Unsloth framework with GRPO. We used a couple different reward functions including closeness to the answer (numerically). However, this doesn’t get down to the crux of the issue which is understanding how the word problem relates to the process, and how the process relates to the answer. Thus, the model wasn’t learning anything. 

Reviewing the literature, we decided to switch over to a symbol-based approach, trying to predict the formula instead of the answer. There were a number of preparations we needed to make to do this. First, we cleaned the data by extracting the numerical answer from each row. Then, we tried to process the data by fixing the formatting to be consistent as well as adding some of our own formatting (such as spacing). We did this to help limit the errors the model could make, like concatenating ‘(‘ to the end of each operator since an operator is always followed by a ‘(‘. We then used this new data with the NLS to help train the GRPO Reinforcement Learning model. After realizing that the Llama model might be a little large for our dataset, we swapped over to the transformer model with cross-entropy loss on each token - essentially, how confident was the model on each particular token and was it correct or not. We also trained on smaller (not necessarily simpler) questions, to reduce the size of the transformer’s matrix. The quicker training speed of the smaller transformer models allowed us to play with the hyperparameters more.

Once the output formula is obtained, we use regex to put it in symbolic form and then call Sympy to turn it into a final answer.


## Validation Methods

We check the accuracy of the models on a sample of a test set, essentially creating an SAT-type exam out of our test set. We also calculate the average NLS across the sample to compare models with each other.

##mResource Requirements and Constraints

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

The number-predicting GRPO model essentially had an accuracy of 0. As a side note, we mistakenly displayed incorrect accuracies in the final presentation as there were a couple of crucial mistakes made in the calculations. The NLS scores, however, remain the same. The formula-predicting GRPO model did better, as it actually seemed to learn the context behind the questions. It’s NLS averaged out to about 0.50 and the accuracy was significantly lower at around 0.05 (both have baseline ~0). This goes back to the issue of exactness - with many tokens in the answer, and an average NLS of 0.50, the chance that all the tokens are completely correct is very slim. Our final FLAN-T5 model performed better. Its average NLS was >0.80, and the accuracy was 0.44 (baselines of ~0.2 and ~0). While we would’ve liked to see higher accuracy, the results show that the FLAN-T5 model was making the right connections between word problem and formula.


## Test Case Results

[INSERT EXAMPLE OF FLAN T5 WORKING ON A QUESTION AND NOT WORKING ON A QUESTION. MAYBE ALSO TALK ABOUT THE INFINITE GENERATION THING WITH GRPO]


## Current Limitations

The GRPO model’s limitations mostly have to do with training speed. Even using the A100, a single step takes almost a minute and you need multiple hundreds of steps to see anything at all. This is an issue because each hour on an A100 is roughly 7.5 compute units, so each training cycle is about $4. There is a lot of experimentation to be done with parameters and it wasn’t feasible with the slow speed of training the GRPO model. We also tried the T4 which has negligible cost, but the training times more than doubled compared to A100 which is even more infeasible.

FLAN-T5 alleviates this a bit. It’s trainable on T4, although the decreased VRAM means the batch size must be reduced when compared to A100. On T4, a typical training cycle takes roughly an hour and change, meaning that trying different hyperparameters is a viable strategy. With A100, the training cycle is not only reasonable but quite quick, with a full cycle being able to be completed in roughly half an hour. The limitations with FLAN-T5 seem to be more with the power of the model and the training data, and less with the ability of the model to train. 


## Resource Usage Measurements

For the GRPO model, VRAM is high but not maxed out on the default settings (between 20 and 35 GB).

By adjusting the batch size for the transformer model, the VRAM can be brought fairly high. On the A100, VRAM usage generally sits between 25 and 30 GB on batches of size 32 with 2 gradient accumulation steps. This can be toned down to fit on a T4 without issue, albeit at a slower training speed.


## Unexpected challenges

The first unexpected challenge we had was trying to get a reward function for the GRPO model that was trying to just predict the right answer. Also, the training time was an unexpected challenge. We knew it would take a while, but we figured that with relatively short inputs and outputs it wouldn’t be too excessive. 

For the GRPO model predicting formulas, the main unexpected challenge we faced was the issue of repeated tokens and no end of sequence (EOS) token. When this happened the model would generate part of a formula, and then it would just repeat a token until the artificial limit (most commonly ‘multiply’). 

The T5 model was a lot smoother, but there were still some unexpected challenges. One was getting the training loop set up, as it comes with a lot of different parts. Its NLS score average was quite high, but the accuracy was much lower. This reflects a model which is getting the general idea, but lacks the weighting to be exact. 

# Next Steps

## Immediate Improvements

At the current stage of the project, there are not really immediate improvements that need to be made. Improvements at this point are likely long-term problems that have many different steps.

## Current Technical Challenges, Questions, and Alternative Approaches

For future work, going back and taking a closer look at the data could be a good idea. With the size of the dataset, it will take an unreasonable amount of time to clean by hand. We did some basic cleaning, but perhaps with more time we could create automated systems that can quickly fix a lot of the issues with the data. At this point, we think that the data quality is hampering the generalizability of the model a little bit, especially as far as grammar is concerned. 

Another technical challenge to be addressed has to do with exactness discussed earlier. We are curious as to whether the last 20% missing in the average NLS score is due to model complexity or training parameters. There are two other versions of reasonably sized FLAN-T5 models, those being small (80M) and large (780M). It would be interesting to compare the overall performance of those models with the base (250M).

## What We Learned

This project was a valuable experience for all of our group. As a collective team, we had not really worked with this type of Machine Learning before, and especially not with optimizing text-to-text transformers and LLMs. Specifically, we learned about GRPO as a concept as well as how tuning different parameters and loss functions affects model training. Despite the accuracy results not being ideal, the final NLS did get quite high, showing that our work worked. Overall, this project was a very rewarding experience for all of us.
