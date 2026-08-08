# IDEA.md — personalised soothing

The idea the policy layer implements: every baby has a *taste* in motion, the
cradle does not know it, and one machine lives with one baby — so it can
afford to learn.

## The premise

The baby has a fixed hidden temperament: it loves some motions, hates others,
and even prefers certain motion *transitions*. The decision layer never sees
that temperament. All it observes is the baby's visible state, moment to
moment, ordered as a happiness scale that acts as the reward signal:

```
happy > fuss > cry-1 > cry-2 > inconsolable
```

A soothing episode then reads like this: the baby fusses, the brain tries a
motion, the state *worsens* — the brain recognises it is moving away from the
goal, revises its strategy and tries another — the state improves, the brain
learns that this motion works for this baby, the baby settles, the brain
commands stop.

Because it is one machine per baby, this is personalised learning: after a
few nights the cradle already knows what works and settles the baby quickly.

## Where each piece landed in the code

| Idea | Implementation |
|---|---|
| The hidden temperament | `perception/baby.py::Personality` — feature-level tastes (shape/size/speed/vibe), id-level love/hate, habituation (motions wear out with use) and a mood cycle. Enabled with `serve.py --baby --personality`; retuned live from the dashboard (`/taste`) |
| The happiness scale | happiness ranks in `core/policy.py` — the outcome a step is credited with |
| The brain that only sees state | `SoothePolicy` with interchangeable brains: `reflex` (local heuristic), `dream` (the DREAM-Chunk planner, `docs/DREAM-CHUNK.md`), `ollama` / `claude` (LLMs). The brain only ever *advises*; `CradleMachine` validates every pick and keeps all safety decisions |
| The taught scenario corpus | `tools/make_scenarios.py` → `data/scenarios.jsonl` — sessions the Claude brain gets few-shot |
| Held-out validation | present a sequence not in the corpus and check which motion the brain commands; measured end-to-end by `tools/learn_report.py` (learning policy vs fixed ladder, same baby) and `tools/state_figure.py` |
| The taste-selection webapp | the dashboard: pick the baby's taste, watch the decision feed and the learned-preferences panel respond |
| The baby image that vision reads | `web/baby.js` draws the state as the Nubzuki mascot on the cradle iPad; `perception/nubzuki.py` reads it back through a camera |

## Appendix — the original brainstorm (translated)

Kept as written on day 2; the table above is what became of it.

> Teach motions 1–10. Our baby had a fixed personality — loves motion 5,
> completely rejects 3 and 7, likes going from motion 2 to motion 4.
>
> But the LLM does not know our baby's personality. It only receives, moment
> to moment, the expression / whether the baby is happy or not.
>
> So we generate scenarios of the baby's reaction sequences.
>
> happy > cry-1 > cry-2 > cry-3 > unhappy — this is the happiness order the
> LLM perceives; effectively the reward.
>
> Example: starts happy → cry-2 (the LLM tries motion 3) → jumps to unhappy
> (the LLM notices it has left the goal and decides to revise its strategy →
> tries motion 5) → cry-1 (the LLM learns that 5 is good) → happy (the LLM
> commands stop).
>
> We create many such scenarios and teach them to the LLM. To check the
> learning: present a sequence that is *not* in the scenarios and see which
> motion the LLM commands.
>
> For the presentation — "aren't you only raising one baby?" — it is one
> machine per baby; the learning is personalised — even better — after five
> nights it soothes the baby right away.
>
> Webapp: the user selects the baby's taste in motion → a motion plays (or
> sim) → the LLM agent or another algorithm changes the motion to fit the
> baby's feeling.
>
> Computer vision: input is image and audio. A tablet image or a virtual
> baby → a baby-image changer → like the webapp circle thing.
