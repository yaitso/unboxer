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

# BONUS: second idea i had no time to implement

having worked a bit on AI Safety™ i kinda have exceptional (almost pliny like) ability to prompt around guardrails and push frontier LLMs to their limits.

and consistently i have found one major hole of them not being able to do tool use heavy (200+ calls) research tasks.

one example in particular i have found none of the models are able to answer systematically is:

> give me a list of all countries that have no rivers ordered by population desc, wadis don't count — water body has to be flowing 24/7/365 to be considered a river.

you can try this yourself: claude, gpt-5, grok, gemini all will give different answers.

and here lies the idea: we can create an RL environment out of this. setup is similar to `unboxer`:

1. adversarial LLM generates a question that is research heavy and requires extensive tool use like example above
2. code LLM generates wikidata SPARQL query that answers this question
3. once executed we have our Q&A pairs to train RL agent on
4. then RL agent is rolled out in sandbox with internet access that filters out wikidata/wikipedia GETs and entirely blocks all POSTs too

   - i guess just like in `unboxer` full linux vm will be the best option here
   - we can adaptively adjust task complexity as it progresses by just generating more complex SPARQL queries that will require more tool calls to gather answers from the internet
   - since the answer set is a table another advantage is that we can do set intersection by primary keys there and assign partial rewards `intersection_rows / total_rows`
5. goto step 1

the question generation via adversarial LLM is also rather straightforward:
- sample random N entities from wikidata
  
  - if not random: sample one random entity and using that entity's human readable name and description perform vector search for similar entities and sample N of them

- query from wikidata a list of `(human_readable_name, entity_id, properties)` for each of the N entities
- given that table and golden set of very creative table samples and corresponding natural language questions we ask:
  
> generate a question that combines as much of {sampled_entities} and their {sampled_properties} as possible in one single sentence query that can after be easily converted into SPARQL query that answers the question correctly, examples: {golden_set_examples}

complexity knobs here are similar:
1. num_entities: number of entities to sample
2. num_properties: number of properties to sample per entity

that's kinda it. should be also infinitely-ish scalable.

some obvious extensions:

1. procure more data and fill your triplestore with it (just read [AKBC archive](https://akbc.pubpub.org/) to learn how to do that)

  - yolo download just like zuck did entirety of libgen/scihub/arxiv/biorxiv/etc
  - do OCR and convert all that to some unstyled but actually semantic HTML (idk ask frontenders to create shadcn for papers)
  - maybe also get some [diffbot.com](https://diffbot.com/) enterprise plan to get even more data