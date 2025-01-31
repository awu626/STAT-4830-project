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

## Basic performance metrics

## Test case results

## Current limitations

## Resource usage measurements

## Unexpected challenges

# Next Steps

## Immediate improvements needed

## Technical challenges to address

## Questions you need help with

## Alternative approaches to try

## What you've learned so far
