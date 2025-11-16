some context i have a take home exercise where i need to develop an RL environment for LLMs, i did come up with an idea that i think is brilliant it is a _code game environment_

at the start of episode adversarial LLM (advLLM) generates some python program we call `blackbox` like

```python
def blackbox(a: float, b: float) -> float:
    return a + (b*sin(a))**2
```

this function then is wrapped into static template we have that we will use for evaluation within stateless sandboxed python environment

advLLM also generates corresponding hypothesis spec eg for function above:

```
spec = {
      "a": st.floats(-10, 10),
      "b": st.floats(-10, 10)
  }
```

we sample give that spec a list of N kwarg dicts for the function to evaluate and run these inside sandbox while capturing their output

then we zip input kwargs list and output dicts list we got together and get IO pairs as few shot context then advLLM samples N+1 kwarg input dict, computes N+1 output and here game begins!

we start RL episode and ask trainLLM 

"""
this is an interactive reverse engineering game playing which you will be trained with RL
at the start of a rollout you have budget of {MAX_TURNS} turns
each you time you call tool it is decremented by 1
if it reaches 0 episode is terminated and you get 0 reward
if you do guess before your budget reaches 0 then your reward is leftover budget
so keep track of budget you have left to ensure you will be able to call <submit>

if you fail to predict proper function and output N+1 we tell you so in tool response, provide actual output value, and give new one N+2, and we repeat that algo until rollout ends

given the list of {N} input-output pairs: {IO_PAIRS}
your task is to reverse engineer blackbox function that produced them and predict output for {N+1}th input: {N_PLUS_ONE_INPUT}

you have access to 2 tools:
1. sandbox within which we execute your code 
<sandbox>
<fn>function to be ran</fn>
<kwargs>dict of kwargs</kwargs>
<kwargs>can call at the same time multiple kwargs to get multiple outputs</kwargs>
</sandbox>
and
2. submit solution which you call from time to time to check after messing around several times with <sandbox> tool and testing you hypothesis functions you think you have a pretty good guess:
<submit>
<fn>function hypothesis</fn>
<output>output for input N+1</output>
</submit>
```