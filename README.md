# rewardly

_somewhat_ theoretically grounded RL environment idea. couple months back, while being very ARC-pilled and Chollet-maxxed, i came up with an alternative definition of intelligence.

## skills are programs

in his seminal paper ["On the Measures of Intelligence"](https://arxiv.org/abs/1911.01547) François says:

> intelligence as skill-acquisition efficiency

let's think a bit deeper tho: what's a _skill_? for example i know how to cook _amazing_ omelette. a skill here is just a fuzzy program i've learned sample-efficiently and my body is merely a substrate for executing it.

so i was thinking: learning a skill is reverse engineering a blackbox program. 


# program synthesis

there's a shitload of literature on program synthesis. but for some reason my cursory glance through it didn't reveal anyone scaling this up to modern LLMs. it's pathetically small models with less than 1B params.

# BONUS: how to fathom "300 IQ AGI"

people throw around 300 IQ like it's incomprehensible. but it's just arithmetic:
the IQ scale is defined such that 16 points = 1σ.
collect human performance data on your task, calculate μ and σ.
a model scoring at μ + 12.5σ is your 300 IQ agent. boom.

the hard part isn't the math — it's finding tasks that actually measure intelligence. i lowkey believe the one presented above gets closest to that if like me you've stared into the psychometrics abyss long enough.

rule of thumb: once AI performance shifts the distribution from normal to bimodal, then gg — we have AGI.