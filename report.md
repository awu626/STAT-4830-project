# Problem Statement

## What are you optimizing? (Be specific)
We are optimizng the location of movable concession stands within existing stadiums in order to maximize revenue based on crowd behavior.

## Why does this problem matter?
The sports industry is huge, and concessions are a non-negliable part of the revenue taken in by venues. By optimizing the location of the concession stands for different games, venues can increase their profit margins.

## How will you measure success?
We will measure success by comparing our simulation data to data we can obtain from actual venues. We can fine-tune our simulation parameters based on current layouts and known revenue, and then we can try to optimize given the same crowd-behvaior parameters.

## What are your constraints?
Our simulation will carry a couple constraints due to the inherent difficulty in predicting human behavior and also lack of readily available data to help fine-tune our simulation. We will have to abstract human behavior down to a few classes. While this is a big constraint, we are reasonably confident it is possible given we have seen it done before.

## What data do you need?
We will need data on how humans behave at sports games, as well as how much they spend and the prices at those games. This will help us get our simulation tuned to the actual human behavior.

## What could go wrong?
This is a fairly novel project for our team as we have not worked with simulations before. We anticipate some time needed to get accustomed to it and to tune our simulations. Human behavior is just something that is difficult to model. Additionally, data may be hard to obtain. The layout of certain stadiums may also be something that is difficult to model, especially if they contain many floors of concessions. Obviously, there are a lot of intricate details that we cannot model within a reasonable timeframe. Our hope is that these have just a small impact on overall behavior.


# Technical Approach

## Mathematical formulation (objective function, constraints)
Our objective function is the revenue generated across all concession stands. It's subject to the constraints of human behavior (how much people are willing to pay, walk, etc).

## Algorithm/approach choice and justification
We are going to use a simulation to model behavior. This will allow us to simulate the situation and get data for different configurations. It will also allow us to run multiple trials fairly quickly, something that will be instrumental in finding the best configurations.

## Implementation strategy
Our implementation will likely consist of an array with objects (walkway, concession stand, seat, obstacle, etc) in which rational agents navigate around. Initially, we will model behavior with simple models, but as our experience levels grow we can get more intricate and detailed. The agents will be able to explore, move around, and interact with things around the map.

## Validation methods
As mentioned above, we will validate based on the availability of real-world data. By replicating stadiums, we can adjust our human behavior models until the numbers start to look like the real values. Then, we can change configurations with the same parameters to find optimal arrangements.


## Resource requirements and constraints
We are not sure how long this will take as the array size grows, and how long each simulation will need to run for. We will most likely run reduced-people simulations compared to real life, that way we don't need to model behaviors of 60,000+ people. We will just have to tune things like balking threshold in order to make sure lines stay reasonable.


# Initial Results

## Evidence your implementation works
We were able to run a very small-scale simulation featuring a small amount of agents, a couple rewards, and a couple obstacles. The agents are able to freely move around outside the obstacles and collect reward points (revenue) by visiting the concession stands.

![Grid Traversal Simulation](simulation.gif)


## Basic performance metrics
We were able to run 10 simulations of 100 timestamps quickly with our small-scale implementation, and figure out the best configuration among the 10 random configurations based on total revenue.

## Test case results
FILL IN LATER

## Current limitations
This is obviously an extremely simple implementation as basically a proof of concept. It does not consider obstacles, actual behvaior (you would want to watch the game, not walk around aimlessly), doesn't include prices, and only uses 1 agent.

## Resource usage measurements
At this small scale, it is impossible to notice the strain on a modern CPU. As we scale up to a bigger stadium and more agents, we will have to come back and revisit this section.

## Unexpected challenges
Most of the challenges we have faced so far are expected. The only unexpected challenge we have faced so far is realizing that modeling human behavior might be even more difficult that we initially thought.

# Next Steps

## Immediate improvements needed
We need to start modeling human behavior soon. Instead of us having rewards, tiles, and obstacles, we need to create objects for these with different properties as well as seats and other stadium components. This will allow us to further evaluate the feasibility of this project idea in modeling actual human behavior.

## Technical challenges to address
While we see no signs of strain on our small implementation, it is an open question whether our local machines will hold up to thousands of simulations of thousands of agents in large stadiums. We may have to streamline our code to make it more efficient at large scales. We may have to leverage GPUs, although we would need to find an optimized framework that supports simulation and GPU usage (maybe CuPy).

## Questions you need help with
We mainly need help finding the real-life data numbers that will help us validate our simulations. This is not information that is readily available, so we will have to work with what we're given.

## Alternative approaches to try
While this project itself does not have alternative approaches, we have considered doing this simulation in different scenarios that may be easier to do. An example of another project in this category would be modeling the most optimal way to board or deboard an airplane. There are a lot less behavior models and variables to take in to model that, and it would still keep the core idea and be an important problem to address.

## What you've learned so far
We have learned that modeling human behavior will be more difficult than original anticipated. However, the idea shows some promise provided that the neccessary data can be obtained.
