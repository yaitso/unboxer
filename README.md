# unboxer

_somewhat_ theoretically grounded RL environment idea.

couple months back, while being very ARC-pilled and Chollet-maxxed, i came up with an alternative definition of intelligence.

## skills are programs

in his seminal paper ["On the Measures of Intelligence"](https://arxiv.org/abs/1911.01547) François says:

> intelligence as skill-acquisition efficiency

what did he mean by **skill**? we may never know.

but for me i think of it this way: i know how to cook amazing omelette — **skill** here is just fuzzy program i've learned sample-efficiently and my body is merely a substrate for executing it while brain is a storage of it for future use.

so i was thinking: if learning a skill is simply reverse engineering of an unknown program, why not build a RL environment for this?

presenting to you ✨ **unboxer** ✨

this environment is infinitely scalable. meaning you can always generate bigger, more complex programs that the model will be asked to unbox via tool use during training. we can adaptively adjust task complexity as it progresses: you need mean solve rate of .4? sure we ask our adversarial LLM: given a `list[(fn_src, fn_mean_reward)` over sliding window of `{N}` episodes and taking into account `{current_solve_rate}` plz adjust diffulty of next batch's _new_ functions you generate to achieve `{target_solve_rate}` in sliding windows of subsequent episodes. this way we can keep the model always challenged and learning.

# game setup

<!-- TODO -->

# program synthesis

there's a shitload of literature on program synthesis. but for some reason my cursory glance through it didn't reveal anyone scaling this up as i did as objective of RL training of modern LLMs. plz hire me my family is starving.


# future work

- functions with hidden state ~ POMDPs
- autoregressive functions ~ bit easier than POMDPs cuz you can at least see that hidden state
- more tools:
  - VLMs will probably be more sample-efficient via eyeballing of .png files of function plots
- perf & stability

# BONUS: how to fathom "300 IQ AGI"

people throw around 300 IQ like it's incomprehensible. but it's just arithmetic:
the IQ scale is defined such that 16 points = 1σ.
collect human performance data on your task, calculate μ and σ.
a model scoring at μ + 12.5σ is your 300 IQ agent. boom.

the hard part isn't the math — it's finding tasks that actually measure intelligence. i lowkey believe the one presented above gets closest to that if like me you've stared into the psychometrics abyss long enough.

rule of thumb: once AI performance shifts the distribution from normal to bimodal, then gg — we have AGI.