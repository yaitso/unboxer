# unboxer

infinitely scalable RL environment for program synthesis via reverse engineering.

## motivation

couple months back, while being very ARC-pilled and Chollet-maxxed, i came up with an alternative definition of intelligence.

in his seminal paper ["On the Measure of Intelligence"](https://arxiv.org/abs/1911.01547) François says:

> intelligence as skill-acquisition efficiency

what did he mean by **skill**? we may never know.

but for me i think of it this way: i know how to cook amazing omelette — **skill** here is just fuzzy program i've learned sample-efficiently and my body is merely a substrate for executing it while brain is a storage of it for future use.

so i was thinking: if learning a skill is simply reverse engineering of an unknown program, why not build an RL environment for this?

## how it works

### adaptive curriculum via adversarial LLM

most curriculum RL uses heuristic difficulty knobs. boring and doesn't scale.

instead: adversarial LLM generates functions at target difficulty based on recent solve rates.

1. track agent performance across rolling window (last 100 functions)
2. solve rate too high? → generate harder functions (more ops/holes/args)
3. solve rate too low? → dial it back
4. agent stays perpetually challenged

this is infinitely scalable — you can always cook up more complex programs. there's a shitload of literature on program synthesis but for some reason nobody scaled this up as RL objective for modern LLMs.

### sandboxed execution

each rollout spins up isolated fly.io VM:
- docker container (python + math libs)
- rust SSH proxy (connection management + auto-shutdown after idle)
- persistent workspace volume

agent gets three tools:
- `bash(cmd)`: run shell commands, install packages, write scripts
- `eval(fn, kwargs)`: test function hypotheses in isolation
- `submit(fn, output)`: claim you solved it

reward = budget remaining at solve (starts at `max_turns`, decrements each turn). this encourages sample efficiency!

### postgres tracking

postgres tracks everything (rollouts, trajectories, tool calls, solve rates). sliding window queries power curriculum decisions. schema is straightforward:

```sql
CREATE TABLE unboxer.rollouts (
    blackbox TEXT,          -- target function
    reward REAL,            -- budget remaining at solve
    solved BOOLEAN,
    trajectory JSONB,       -- full history
    logs JSONB,             -- stuff we wanted to track for debugging
    ...
)
```

### training

modal H100 → vLLM inference → training via verifiers → push to 🤗 hub

```bash
un setup    # create modal volume (once)
un build    # docker image → fly.io registry
un train    # yolo H100
```

config lives in `configs/unboxer.toml`, tweak as needed.

## implementation notes

**function generation**: haiku generates functions matching exact complexity (num_ops, num_holes, num_args). prompts have strict counting rules to keep it honest.

**complexity knobs**:
- num_ops: operators + math functions (sin, exp, etc)
- num_holes: constants to discover ($a$, $b$)  
- num_args: function arity

## status

done-ish in ~25h (five of which were spent fighting flash attention build lmao).

full pipeline works: blackbox function generation, sandbox infra, postgres tracking, modal training, HF uploads. initial training runs completed tho didn't scale cause GPU poor 😭.

## future work

- **deterministic shape enumeration**: initially wanted to build greedy BFS traversal of all possible AST nodes (valid programs by construction, also had a way to ensure totality/termination but forgot how lmao). would cache + reuse subtrees when building higher-depth ASTs. then i went "wait this is a side quest for MVP, i can just ask claude to generate more complex functions" — like yeah it uses billion times more FLOPS than rust shape enumeration but hey we're in 2025, it's aight. still, proper enumeration would be cleaner for curriculum guarantees.
- functions with hidden state ~ POMDPs
- autoregressive functions ~ bit easier than POMDPs cuz you can at least see that hidden state
- VLM tools for plot visualization (probably way more sample-efficient — remember recent DeepSeek OCR?)
- actual scale up on big chungus frontier lab cluster?

## BONUS: how to fathom "300 IQ AGI"

people throw around 300 IQ like it's incomprehensible. but it's just arithmetic:

IQ scale is defined such that 16 points = 1σ. collect human performance data on your task, calculate μ and σ. a model scoring at μ + 12.5σ is your 300 IQ agent. boom.

the hard part isn't the math — it's finding tasks that actually measure intelligence. i lowkey believe program synthesis via reverse engineering gets closest to that if like me you've stared into the psychometrics abyss long enough.

rule of thumb: once AI performance shifts the distribution from normal to bimodal, then gg — we have AGI.

**why is this a good proxy for intelligence?**

think about what humans actually do when they understand the world:

**physics**: we sample data (experiments) and fit functions that correctly predict OOD inputs. one great feature of our universe is we have pockets of computational reducibility — we don't simulate every particle, we find the compact programs (F=ma, etc etc) that generalize.

**biology**: simulating molecular interactions is infeasible (whole planet is too GPU poor for that), but we still do GWAS, interventional experiments, etc. we're basically saying "idk which exact knobs underlie this trait, but with enough data we can say *this set* of alleles explains N% of variance" or "downregulating this gene → that phenotype with K% certainty". that's reverse engineering a probabilistic function!

**basically everything**: if you can't reverse engineer the programs underlying observed behavior, you can't compress reality into useful abstractions. you're stuck memorizing lookup tables.

**also unlike ARC AGI**: we don't have to painstakingly craft examples by hand. since we intend to cover all total (terminating) programs in our generation process, the ARC AGI problems are mathematically speaking *enclosed within* the space we sample functions from. if you do systematic shape enumeration (see future work below), you'd eventually generate every valid ARC puzzle as a special case.

## BONUS: second idea i had no time to implement

having worked a bit on AI Safety™ i kinda have exceptional (almost pliny like) ability to prompt around guardrails and push frontier LLMs to their limits.

and consistently i have found one major hole of them not being able to do tool use heavy (200+ calls) research tasks.

one example in particular i have found none of the models are able to answer systematically is:

> give me a list of all countries that have no rivers ordered by population in descending order, wadis don't count — water body has to be flowing 24/7/365 to be considered a river.

you can try this yourself: claude, gpt-5, grok, gemini all will give different answers.

and here lies the idea: we can create an RL environment out of this.

**setup is similar to unboxer:**

1. adversarial LLM generates research-heavy question requiring extensive tool use
2. code LLM generates wikidata SPARQL query that answers it
3. execute query → we have ground truth Q&A pairs
4. RL agent rolls out in sandbox with internet access
   - filter out wikidata/wikipedia GETs (no cheating)
   - block all POSTs entirely
   - full linux VM like in unboxer
5. adaptive difficulty: generate more complex SPARQL queries as agent improves
6. partial credit via set intersection: `reward = matching_rows / total_rows`
7. goto step 1

**question generation mechanics:**

sample random N entities from wikidata. optionally: instead of random, do vector search for similar entities based on first sample's description.

query wikidata for `(human_readable_name, entity_id, properties)` for each entity.

given that table + golden set of creative examples, prompt:

> generate a question that combines as much of {sampled_entities} and their {sampled_properties} as possible in one single sentence query that can be converted into SPARQL query answering it correctly. examples: {golden_set_examples}

**complexity knobs:**
- num_entities: how many entities to combine
- num_properties: properties per entity
- query execution FLOPS: might be good proxy for complexity

**obvious extensions:**

procure more data for triplestore (read [AKBC archive](https://akbc.pubpub.org/) to learn how):
- yolo download entirety of libgen/scihub/arxiv/biorxiv (just like zuck did)
- OCR everything → semantic HTML (idk ask frontenders to create shadcn for papers)
- diffbot.com enterprise plan for web-scale data

that's kinda it. should be infinitely-ish scalable too.

---

plz hire me my family is starving.