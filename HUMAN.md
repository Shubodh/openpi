# Step 4 Decision Framework

## Questions for Step 4 Scope

### A) Patching Implementation Architecture

Where should the patching code live, and how modular should it be?

- Option 1: New file `examples/libero/main_patching_expt.py` — clean separation, full control
- Option 2: Extend `main_corrupt_run_expt.py` with a `--args.enable-patching` flag — reuses infrastructure
- Option 3: A standalone intervention layer/utility that can be called from either script

**Your choice:** 

This is an important question. So far, I have done SmolVLA patching experiments for real world setting in `/home/shubodh/claude_code_workspace/2026_AXMech/AXMech` folder. For more context, you can go through the CLAUDE.md of that particular folder. The patching was successful, and I think for SmolVLA, it is easier than pi0.5.  

I definitely don't want to extend the `main_corrupt_run_expt.py` since that would make the code messier and harder to debug.

Can you go through the patching code I have written for SmolVLA and let me know if can follow a similar structure? The question is open - whether we want to do it in AXMech repo or here. Even if we decide here, how exactly is also still an open question. 



---

### B) KV-Cache Intervention Mechanism

Before writing patching code, do we do the KV-cache sanity check first, or can the agent infer the strategy from the SO-101 design and proceed?

- Option 1: You run the sanity check (inspect cache shapes, token positions) now, then brief the agent on findings
- Option 2: Agent does the sanity check as part of implementation (slower, but self-contained)
- Option 3: Agent writes the patching code assuming SO-101 pattern, then does sanity check to validate

**Your choice:**
Let's go with Option 1. 

---

### C) Experiment Workflow

Should the final script run all three conditions (clean/corrupt/patched) in one go, or separate?

- Option 1: Single script with three CLI modes (`--mode clean|corrupt|patched`)
- Option 2: Three separate invocations (reuse existing infrastructure, easier to debug independently)

**Your choice:**
We need to make it similar to AXMech implementation.

---

### D) LIBERO-Goal Baseline First

Before patching, should we confirm LIBERO-Goal baseline works (quick 30-min test with a few tasks)?

- This would validate that the suite generalizes well (unlike LIBERO-Object)
- Gives us a sanity check that the model can learn these contrastive pairs

**Your choice:**
This is quite fast, I can do it on my end. You dont need to worry about it.

---

## Context Reminder

- **Task pair for first experiment:** `put_the_bowl_on_top_of_the_cabinet` vs. `put_the_wine_bottle_on_top_of_the_cabinet`
- **Suite:** LIBERO-Goal (identical layout, contrastive prompts differ by one object word)
- **Goal:** One end-to-end patching experiment validating the mechanism
- **Mechanism:** KV-cache intervention at object-name token position (based on SO-101 pattern)
